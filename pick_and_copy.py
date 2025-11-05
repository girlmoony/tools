# pick_and_copy.py
import argparse
import io
import json
import os
import re
import sys
import time
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque

import pandas as pd
from tqdm import tqdm

# ========== ファイル名解析 ==========
# 例: [1.00]1304_011-01_2024-09-02_12-33-32-709_1.png
FNAME_RE = re.compile(
    r"""
    ^\[
        (?P<score>\d+(?:\.\d+)?)
    \]
    (?P<store>\d+)
    _
    (?P<lane_block>\d{2,3}-\d{2,3})
    _
    (?P<date>\d{4}-\d{2}-\d{2})
    _
    (?P<time>\d{2}-\d{2}-\d{2}-\d{3})
    _
    (?P<seq>\d+)
    \.(?P<ext>png|raw)$
    """,
    re.VERBOSE | re.IGNORECASE,
)

def parse_filename(name: str):
    m = FNAME_RE.match(name)
    if not m:
        return None
    d = m.groupdict()
    # レーン番号の扱い: "011-01" の前半をレーン、後半をサブ(カメラ)として保持
    lane_part = d["lane_block"].split("-")
    lane = lane_part[0]  # '011'
    cam  = lane_part[1]  # '01'
    return {
        "score": float(d["score"]),
        "store": d["store"],
        "lane": lane,
        "cam": cam,
        "ext": d["ext"].lower(),
    }

# ========== FS 抽象化 (Local / SFTP) ==========
class FSBase:
    def listdir(self, p): raise NotImplementedError
    def isfile(self, p): raise NotImplementedError
    def isdir(self, p): raise NotImplementedError
    def exists(self, p): raise NotImplementedError
    def makedirs(self, p, exist_ok=True): raise NotImplementedError
    def open_rd(self, p): raise NotImplementedError
    def open_wr(self, p): raise NotImplementedError
    def copyfile(self, src_fs, src_path, dst_path): raise NotImplementedError
    def join(self, *a): return "/".join(str(x).rstrip("/").lstrip("\\") for x in a).replace("//","/")
    def norm(self, p): return p

class LocalFS(FSBase):
    def listdir(self, p):
        return os.listdir(p)
    def isfile(self, p):
        return os.path.isfile(p)
    def isdir(self, p):
        return os.path.isdir(p)
    def exists(self, p):
        return os.path.exists(p)
    def makedirs(self, p, exist_ok=True):
        os.makedirs(p, exist_ok=exist_ok)
    def open_rd(self, p):
        return open(p, "rb")
    def open_wr(self, p):
        parent = os.path.dirname(p)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        return open(p, "wb")
    def copyfile(self, src_fs, src_path, dst_path):
        if isinstance(src_fs, LocalFS):
            parent = os.path.dirname(dst_path)
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)
            shutil.copy2(src_path, dst_path)
        else:
            # src: remote / dst: local
            with src_fs.open_rd(src_path) as r, self.open_wr(dst_path) as w:
                shutil.copyfileobj(r, w)

class SFTPFS(FSBase):
    def __init__(self, host, user, password=None, port=22, key_filename=None):
        import paramiko
        self._client = paramiko.SSHClient()
        self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self._client.connect(
            hostname=host, username=user, password=password,
            port=port, key_filename=key_filename, look_for_keys=False, allow_agent=False
        )
        self._sftp = self._client.open_sftp()

    def __del__(self):
        try:
            self._sftp.close()
            self._client.close()
        except Exception:
            pass

    def listdir(self, p):
        return [e.filename for e in self._sftp.listdir_attr(p)]
    def isfile(self, p):
        try:
            st = self._sftp.stat(p)
            # paramiko: S_ISREG は無いので size が取れれば概ねOK とする
            return True
        except IOError:
            return False
    def isdir(self, p):
        import stat
        try:
            st = self._sftp.stat(p)
            return stat.S_ISDIR(st.st_mode)
        except IOError:
            return False
    def exists(self, p):
        try:
            self._sftp.stat(p); return True
        except IOError:
            return False
    def makedirs(self, p, exist_ok=True):
        parts = [x for x in p.split("/") if x]
        cur = "/" if p.startswith("/") else ""
        for part in parts:
            cur = (cur + "/" + part) if cur else part
            try:
                self._sftp.stat(cur)
            except IOError:
                self._sftp.mkdir(cur)
    def open_rd(self, p):
        return self._sftp.open(p, "rb")
    def open_wr(self, p):
        parent = "/".join(p.rstrip("/").split("/")[:-1])
        if parent:
            self.makedirs(parent, exist_ok=True)
        # 強制的にバイナリ書込み
        return self._sftp.open(p, "wb")
    def copyfile(self, src_fs, src_path, dst_path):
        # remote<-local または remote<-remote の両方をサポート
        if isinstance(src_fs, SFTPFS) and src_fs is self:
            # 同一ホスト：一時ファイルなしコピー（SFTP にはサーバーサイドコピーが無いのでストリーム）
            with src_fs.open_rd(src_path) as r, self.open_wr(dst_path) as w:
                shutil.copyfileobj(r, w)
        else:
            # src: local or 別ホスト
            with src_fs.open_rd(src_path) as r, self.open_wr(dst_path) as w:
                shutil.copyfileobj(r, w)

# ========== ユーティリティ ==========
def build_fs(mode: str, root: str, ssh_args: dict):
    if mode == "local":
        fs = LocalFS()
    elif mode == "ssh":
        fs = SFTPFS(**ssh_args)
    else:
        raise ValueError("--*-mode は local か ssh を指定してください")
    # root は最後の '/' を落とさない
    root = root.rstrip("/")
    return fs, root

def safe_rename_dir(fs: FSBase, root_path: str):
    """出力ルートが存在する場合はタイムスタンプでリネーム"""
    if fs.exists(root_path):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_path = root_path + f"_backup_{ts}"
        # 既にバックアップ名があった場合のため微妙にずらす
        idx = 1
        while fs.exists(new_path):
            idx += 1
            new_path = root_path + f"_backup_{ts}_{idx}"
        print(f"[INFO] Rename dst root: {root_path} -> {new_path}")
        # SFTP にはディレクトリの再帰移動がないので、親で rename を使う（rename は OK）
        # ただし rename が効かない FS の場合はユーザー側で調整してください
        try:
            if isinstance(fs, LocalFS):
                os.rename(root_path, new_path)
            else:
                fs._sftp.rename(root_path, new_path)
        except Exception as e:
            print(f"[WARN] rename 失敗: {e}. バックアップせず上書きします。")

def read_excel_mapping(excel_path: str):
    df = pd.read_excel(excel_path, engine="openpyxl")
    # 期待列: 商品コード / クラス分類 / クラス名
    cols = df.columns.tolist()
    # 緩めに解決
    col_code = next(c for c in cols if "商品コード" in c)
    col_type = next(c for c in cols if "クラス分類" in c)
    col_name = next(c for c in cols if "クラス名" in c)

    # 統合セルがあっても pandas は値を下にコピーしないことがあるため forward-fill
    df[col_type] = df[col_type].ffill()
    df[col_name] = df[col_name].ffill()

    mapping = defaultdict(set)   # class_name -> {product_codes}
    class_rows = []              # クラス一覧（順序保持用）

    for _, row in df.iterrows():
        code = str(row[col_code]).strip()
        ctype = str(row[col_type]).strip()
        cname = str(row[col_name]).strip()
        if not cname:
            continue
        if ctype == "単独クラス":
            if code:
                mapping[cname].add(code)
        elif ctype == "まとめクラス":
            if code:
                mapping[cname].add(code)
        else:
            # 想定外の分類名も許容してコードを結びつける
            if code:
                mapping[cname].add(code)
        class_rows.append(cname)

    # クラス名の順序は Excel の並びを優先
    ordered_classes = []
    seen = set()
    for c in class_rows:
        if c and c not in seen:
            ordered_classes.append(c)
            seen.add(c)

    return ordered_classes, mapping

def load_shopinfo(shopinfo_path: str):
    with open(shopinfo_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 期待： { "1304": {"model": "new"|"old", "lanes": ["01","02",...,"12"]}, ... }
    # フォーマットが異なる場合はこの関数を調整
    norm = {}
    for store, info in data.items():
        model = str(info.get("model", "")).lower()
        lanes = [str(x).zfill(2) for x in info.get("lanes", list(range(1,13)))]
        norm[str(store)] = {"model": model, "lanes": lanes}
    return norm

def find_class_dirs(fs: FSBase, src_root: str):
    """クラスフォルダを列挙（直下）"""
    dirs = []
    for name in fs.listdir(src_root):
        p = fs.join(src_root, name)
        try:
            if fs.isdir(p):
                dirs.append((name, p))
        except Exception:
            continue
    return dirs

def iter_images_in(fs: FSBase, dir_path: str):
    """指定フォルダ直下の .png/.raw"""
    try:
        for name in fs.listdir(dir_path):
            if name.lower().endswith((".png", ".raw")):
                yield name
    except Exception:
        return

def pick_144(files_by_store_lane: dict, wanted_stores: list, lanes_per_store: dict):
    """
    1クラス分・1モデル(new/old)分のファイルプールから 144 枚を選抜。
    ポリシー:
      - まず 各 店舗×各レーン から 1枚ずつ（ラウンドロビン）
      - 144 未満なら、再ラウンドで未使用ファイルを追加（再ループ）
    """
    picked = []
    composition = defaultdict(lambda: defaultdict(int))  # store -> lane -> count
    # 各 (store, lane) ごとにキューを作る
    queues = []
    for store in wanted_stores:
        lanes = lanes_per_store.get(store, [])
        for lane in lanes:
            q = deque(files_by_store_lane.get((store, lane), []))
            if q:
                queues.append(((store, lane), q))

    # ラウンドロビン
    while len(picked) < 144 and queues:
        progressed = False
        new_queues = []
        for (store, lane), q in queues:
            if len(picked) >= 144:
                break
            if q:
                f = q.popleft()
                picked.append(f)
                composition[store][lane] += 1
                progressed = True
                if q:
                    new_queues.append(((store, lane), q))
        queues = new_queues
        if not progressed:
            break  # もう追加できない

    return picked, composition

def ensure_dir(fs: FSBase, p: str):
    if not fs.exists(p):
        fs.makedirs(p, exist_ok=True)

def write_report(out_path: str, rows: list):
    """
    rows: dict のリスト
      { "クラス名": str, "型": "新型"/"旧型", "構成": "店舗XXXX レーンYY：ZZ枚, ..." }
    """
    df = pd.DataFrame(rows, columns=["クラス名", "型", "構成"])
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="summary")

def main():
    ap = argparse.ArgumentParser(description="197フォルダからクラス別に144枚×(新/旧)を選抜してコピー")
    ap.add_argument("--excel-map", required=True, help="クラス定義Excel（商品コード/クラス分類/クラス名）")
    ap.add_argument("--shopinfo", required=True, help="shopinfo.json（店舗コード→新旧/レーン配列）")

    # Source
    ap.add_argument("--src-mode", choices=["local", "ssh"], default="local")
    ap.add_argument("--src-root", required=True)
    ap.add_argument("--ssh-host", help="SSH host")
    ap.add_argument("--ssh-port", type=int, default=22)
    ap.add_argument("--ssh-user", help="SSH user")
    ap.add_argument("--ssh-pass", help="SSH password")
    ap.add_argument("--ssh-key", help="SSH private key file")

    # Dest NEW
    ap.add_argument("--dst-new-mode", choices=["local", "ssh"], required=True)
    ap.add_argument("--dst-new-root", required=True)
    # Dest OLD
    ap.add_argument("--dst-old-mode", choices=["local", "ssh"], required=True)
    ap.add_argument("--dst-old-root", required=True)

    ap.add_argument("--rename-existing-dst-root", action="store_true", help="出力ルートが存在すれば時刻付きでリネーム")
    ap.add_argument("--out-report", required=True, help="構成Excelの出力先")
    ap.add_argument("--prefer-png", action="store_true", help="同じ条件でpng/rawがあればpngを優先")

    args = ap.parse_args()

    # FS 準備
    ssh_args = dict(
        host=args.ssh_host, user=args.ssh_user, password=args.ssh_pass,
        port=args.ssh_port, key_filename=args.ssh_key
    )

    src_fs, src_root = build_fs(args.src_mode, args.src_root, ssh_args if args.src_mode=="ssh" else {})
    new_fs, new_root = build_fs(args.dst_new_mode, args.dst_new_root, ssh_args if args.dst_new_mode=="ssh" else {})
    old_fs, old_root = build_fs(args.dst_old_mode, args.dst_old_root, ssh_args if args.dst_old_mode=="ssh" else {})

    if args.rename_existing_dst_root:
        safe_rename_dir(new_fs, new_root)
        safe_rename_dir(old_fs, old_root)

    ensure_dir(new_fs, new_root)
    ensure_dir(old_fs, old_root)

    # マッピング＆店舗情報
    class_order, class_map = read_excel_mapping(args.excel_map)
    shopinfo = load_shopinfo(args.shopinfo)

    # モデル別の店舗一覧・レーン辞書を用意
    stores_by_model = {"new": [], "old": []}
    lanes_per_store = {}
    for store, info in shopinfo.items():
        model = info["model"]
        lanes = info["lanes"]
        if model in ("new", "old"):
            stores_by_model[model].append(store)
        else:
            # 未指定は両方から外す（必要に応じて調整）
            pass
        lanes_per_store[store] = [str(l).zfill(2) for l in info["lanes"]]

    # 取得対象クラスのフォルダ（src_root直下）を列挙
    src_class_dirs = dict(find_class_dirs(src_fs, src_root))  # {folder_name: path}

    # Excel から出たクラス名のうち、src に存在するものを対象
    target_classes = [c for c in class_order if c in src_class_dirs]

    if not target_classes:
        print("[WARN] Excel のクラス名と一致するフォルダが見つかりませんでした。")
        # 197全部を対象にする場合は次の1行を有効化
        # target_classes = list(src_class_dirs.keys())

    report_rows = []

    # 進捗バー
    for cname in tqdm(target_classes, desc="Processing classes"):
        class_src_dir = src_class_dirs[cname]

        # まずクラスフォルダ直下のファイルを走査し、store/lane でバケット化
        # 並列に png と raw があった場合は prefer-png で優先
        bucket_by_model = {
            "new": defaultdict(list),  # (store,lane) -> [src_path,...]
            "old": defaultdict(list)
        }

        for fname in iter_images_in(src_fs, class_src_dir):
            meta = parse_filename(fname)
            if not meta:
                continue
            store = meta["store"]
            lane  = meta["lane"][-2:]  # "011" -> "11" にせず "11" や "01" も揃える
            lane  = lane.zfill(2)

            model = shopinfo.get(store, {}).get("model", "").lower()
            if model not in ("new", "old"):
                continue

            key = (store, lane)
            fpath = src_fs.join(class_src_dir, fname)
            bucket_by_model[model][key].append((fpath, meta["ext"]))

        # png優先で並べ替え
        if args.prefer_png:
            for model, b in bucket_by_model.items():
                for k, arr in b.items():
                    arr.sort(key=lambda x: (0 if x[1]=="png" else 1, x[0]))

        # モデル別に144枚選抜
        for model in ("new", "old"):
            files_by_store_lane = {}
            for k, arr in bucket_by_model[model].items():
                # 実ファイルパスだけにする
                files_by_store_lane[k] = [p for (p, ext) in arr]

            picked, comp = pick_144(
                files_by_store_lane=files_by_store_lane,
                wanted_stores=stores_by_model[model],
                lanes_per_store=lanes_per_store
            )

            # 出力先クラスフォルダを用意（上書きしない）
            dst_root = new_root if model == "new" else old_root
            dst_fs   = new_fs   if model == "new" else old_fs
            dst_class_dir = dst_fs.join(dst_root, cname)
            ensure_dir(dst_fs, dst_class_dir)

            # コピー（src_fs -> dst_fs）
            for src_path in picked:
                base = os.path.basename(src_path)
                dst_path = dst_fs.join(dst_class_dir, base)
                try:
                    dst_fs.copyfile(src_fs, src_path, dst_path)
                except Exception as e:
                    print(f"[WARN] copy failed: {src_path} -> {dst_path}: {e}")

            # 構成を人間可読にまとめる
            parts = []
            for store in sorted(comp.keys()):
                lanes = comp[store]
                lanes_s = ", ".join([f"レーン{ln}: {cnt}枚" for ln, cnt in sorted(lanes.items())])
                parts.append(f"店舗{store} {lanes_s}")
            summary = " / ".join(parts) if parts else "（選抜なし）"
            report_rows.append({"クラス名": cname, "型": "新型" if model=="new" else "旧型", "構成": summary})

    # レポート出力
    write_report(args.out_report, report_rows)
    print(f"[INFO] 完了。レポート: {args.out_report}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        sys.exit(1)

python pick_and_copy.py ^
  --excel-map C:\data\class_map.xlsx ^
  --shopinfo C:\data\shopinfo.json ^
  --src-mode ssh ^
  --src-root /mnt/sdcard/dataset ^
  --ssh-host 192.168.1.10 --ssh-user ubuntu --ssh-pass yourpassword ^
  --dst-new-mode ssh --dst-new-root /mnt/sdcard/NEW ^
  --dst-old-mode local --dst-old-root D:\OUTPUT\OLD ^
  --rename-existing-dst-root ^
  --out-report C:\data\picked_summary.xlsx


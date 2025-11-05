import pandas as pd
import json
import os
import shutil
import paramiko
import stat
import math
import csv
from collections import defaultdict

# --- 設定情報 ---
EXCEL_PATH = 'class_info.xlsx'
JSON_PATH = 'shopinfo.json'
# ソースパス (リモートまたはローカル)
# ローカルの場合: r'C:\path\to\source\images'
# リモートの場合: '/path/to/remote/sdcard/images'
SOURCE_BASE_PATH = r'C:\Users\YourUser\Desktop\source_images' 
# 保存先ベースパス (リモートまたはローカル)
# ローカルの場合: r'C:\path\to\destination'
# リモートの場合: '/path/to/remote/destination'
DEST_BASE_PATH = r'C:\Users\YourUser\Desktop\destination'

# SSH接続情報 (ローカルパスの場合は不要、リモートパスの場合のみ設定)
USE_SSH = False # リモートパスを使用する場合は True に設定
SSH_HOSTNAME = 'your_remote_host'
SSH_PORT = 22
SSH_USERNAME = 'your_username'
SSH_PASSWORD = 'your_password' # または鍵認証を設定
# -----------------

# SSHクライアントの初期化
ssh_client = None
sftp_client = None

def get_file_manager():
    """ローカルまたはSSH用のファイルマネージャーを返す"""
    if USE_SSH:
        global ssh_client, sftp_client
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(SSH_HOSTNAME, SSH_PORT, SSH_USERNAME, SSH_PASSWORD)
        sftp_client = ssh_client.open_sftp()
        return SFTPFileManager(sftp_client)
    else:
        return LocalFileManager()

class LocalFileManager:
    """ローカルファイルシステム操作のラッパークラス"""
    def listdir(self, path):
        return os.listdir(path)
    def isdir(self, path):
        return os.path.isdir(path)
    def join(self, *args):
        return os.path.join(*args)
    def makedirs(self, path, exist_ok=True):
        os.makedirs(path, exist_ok=exist_ok)
    def copy_file(self, src, dst):
        shutil.copy2(src, dst)
    def exists(self, path):
        return os.path.exists(path)
    def rename(self, src, dst):
        os.rename(src, dst)

class SFTPFileManager:
    """SFTP（SSH）ファイルシステム操作のラッパークラス"""
    def __init__(self, sftp):
        self.sftp = sftp

    def listdir(self, path):
        return [f.filename for f in self.sftp.listdir_attr(path)]

    def isdir(self, path):
        try:
            return stat.S_ISDIR(self.sftp.stat(path).st_mode)
        except IOError:
            return False
            
    def exists(self, path):
        try:
            self.sftp.stat(path)
            return True
        except IOError:
            return False

    def join(self, *args):
        return '/'.join(args).replace('//', '/')

    def makedirs(self, path, exist_ok=True):
        # SFTPでの再帰的なディレクトリ作成
        dirs = path.split('/')
        current_path = ''
        for d in dirs:
            if d:
                current_path = self.join(current_path, d)
                if not self.exists(current_path):
                    self.sftp.mkdir(current_path)

    def copy_file(self, src, dst):
        # SFTPでファイルをコピー（リモートtoリモートは非効率だが、現状sftp.get/putを使うしかない）
        # ここではpythonがsrcをダウンロードしてからdstへアップロードする
        local_temp = f"/tmp/{os.path.basename(src)}"
        print(f"Downloading {src} to {local_temp}...")
        self.sftp.get(src, local_temp)
        print(f"Uploading {local_temp} to {dst}...")
        self.sftp.put(local_temp, dst)
        os.remove(local_temp)
        print("Copy finished.")
        
    def rename(self, src, dst):
        self.sftp.rename(src, dst)

def parse_filename(filename):
    """ファイル名から店舗コード、レーン番号を抽出する"""
    parts = filename.split('_')
    if len(parts) >= 3:
        # [1.00]1304_011-01_2024-09-02... -> 1304_011-01を取得
        shop_lane_part = parts[0].split(']')[-1] + '_' + parts[1]
        
        # 1304_011-01 から店舗コード 1304 と レーン番号 01 を抽出
        shop_code = shop_lane_part.split('_')[0]
        lane_code = shop_lane_part.split('-')[-1]
        return shop_code, lane_code
    return None, None

def get_shop_type(shop_code, shop_info):
    """店舗コードから新型/旧型を取得する"""
    for info in shop_info.values():
        if info.get('店舗コード') == shop_code:
            return info.get('新型/旧型')
    return None

def main():
    fm = get_file_manager()

    try:
        # 2. 保存先フォルダが存在する場合、リネーム（今回は上書きしない設定のためスキップ）

        # 3. ExcelファイルとJSONファイルの読み込み
        df = pd.read_excel(EXCEL_PATH)
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            shop_info = json.load(f)

        # 4. Excelデータの処理
        class_map = defaultdict(list)
        for index, row in df.iterrows():
            prod_code = str(row['商品コード']).strip()
            class_name = str(row['クラス名']).strip()
            if prod_code and class_name:
                class_map[class_name].append(prod_code)
        
        # 5 & 6. 画像のピックアップとコピー
        SHOP_TYPES = ['新型', '旧型']
        summary_data = defaultdict(lambda: defaultdict(int)) # クラス名 -> 店舗コード_レーンNo -> 枚数

        for shop_type in SHOP_TYPES:
            print(f"--- Processing {shop_type} shops ---")
            type_dest_path = fm.join(DEST_BASE_PATH, shop_type)
            fm.makedirs(type_dest_path, exist_ok=True)

            for class_name, prod_codes in class_map.items():
                print(f"  Processing class: {class_name}")
                class_dest_path = fm.join(type_dest_path, class_name)
                # 5. フォルダが存在しなければ作成（上書きしない）
                if fm.exists(class_dest_path) and len(fm.listdir(class_dest_path)) >= 144:
                     print(f"    Folder {class_dest_path} already has enough files. Skipping.")
                     continue
                fm.makedirs(class_dest_path, exist_ok=True)

                # 必要な画像情報を収集
                available_images = []
                for prod_code in prod_codes:
                    source_folder_name = f"{prod_code}_XXXXX" # ここは実際のフォルダ名パターンに合わせる必要があります
                    source_folder_path = fm.join(SOURCE_BASE_PATH, source_folder_name)
                    if fm.exists(source_folder_path):
                        for filename in fm.listdir(source_folder_path):
                            if filename.lower().endswith(('.png', '.raw')):
                                shop_code, lane_code = parse_filename(filename)
                                if shop_code and get_shop_type(shop_code, shop_info) == shop_type:
                                    available_images.append({
                                        'path': fm.join(source_folder_path, filename),
                                        'shop_code': shop_code,
                                        'lane_code': lane_code,
                                        'filename': filename
                                    })
                
                # ピックアップロジック (144枚)
                target_count = 144
                selected_images = []
                used_combinations = set() # (shop_code, lane_code)
                
                # 1回目のループで各店舗・レーン1枚ずつ取る
                for img in available_images:
                    combo = (img['shop_code'], img['lane_code'])
                    if combo not in used_combinations and len(selected_images) < target_count:
                        selected_images.append(img)
                        used_combinations.add(combo)
                        summary_data[class_name][f"{img['shop_code']}_L{img['lane_code']}"] += 1
                
                # 144枚に満たない場合、再ループして取る
                if len(selected_images) < target_count:
                    loops_needed = math.ceil((target_count - len(selected_images)) / len(available_images))
                    for _ in range(loops_needed):
                        for img in available_images:
                             if len(selected_images) < target_count:
                                selected_images.append(img)
                                summary_data[class_name][f"{img['shop_code']}_L{img['lane_code']}"] += 1
                
                # ファイルのコピー
                print(f"    Copying {len(selected_images)} images to {class_dest_path}...")
                for img in selected_images:
                    dest_file_path = fm.join(class_dest_path, img['filename'])
                    # 同名ファイルの上書き防止
                    if not fm.exists(dest_file_path):
                         fm.copy_file(img['path'], dest_file_path)
                print(f"    Finished class {class_name}.")

        # 7. エクセルに出力（CSVファイルとして出力）
        output_csv_path = fm.join(DEST_BASE_PATH, 'image_composition_summary.csv')
        print(f"Writing summary to {output_csv_path}")

        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['クラス名', '構成(店舗コード_レーンNo:枚数)'])
            for class_name, compositions in summary_data.items():
                composition_str = ' | '.join([f"{k}:{v}枚" for k, v in compositions.items()])
                writer.writerow([class_name, composition_str])
        
        print("処理が完了しました。")

    except Exception as e:
        print(f"エラーが発生しました: {e}")
    finally:
        if sftp_client:
            sftp_client.close()
        if ssh_client:
            ssh_client.close()

if __name__ == "__main__":
    main()

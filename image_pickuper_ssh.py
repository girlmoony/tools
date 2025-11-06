import pandas as pd
import json
import os
import shutil
import paramiko
import stat
import math
import csv
import time
from collections import defaultdict

# --- 設定情報 ---
EXCEL_PATH = 'class_info.xlsx'
JSON_PATH = 'shopinfo.json'
# ソースパス (リモートのSDカードパス)
SOURCE_BASE_PATH = '/path/to/remote/sdcard/images' 
# 保存先ベースパス (リモートのSDカードパス)
DEST_BASE_PATH = '/path/to/remote/destination'

# SSH接続情報 
USE_SSH = True # 常にTrue
SSH_HOSTNAME = 'your_remote_host'
SSH_PORT = 22
SSH_USERNAME = 'your_username'
SSH_PASSWORD = 'your_password' # または鍵認証を設定
# -----------------

# SSHクライアントの初期化
ssh_client = None

def run_remote_command(command):
    """リモートSSHサーバー上でコマンドを実行し、出力を返す"""
    if ssh_client is None:
        raise Exception("SSH client is not connected.")
    
    # print(f"Executing: {command}")
    stdin, stdout, stderr = ssh_client.exec_command(command)
    stdout_str = stdout.read().decode('utf-8').strip()
    stderr_str = stderr.read().decode('utf-8').strip()
    exit_status = stdout.channel.recv_exit_status()

    if exit_status != 0:
        raise Exception(f"Remote command failed (Exit code {exit_status}): {command}\nError: {stderr_str}")
    return stdout_str

class SSHRemoteManager:
    """SSHコマンド実行ベースのファイル操作ラッパークラス"""
    def listdir(self, path):
        # ls -F でディレクトリには / をつける
        output = run_remote_command(f"ls -F '{path}'")
        if not output:
            return []
        return output.split('\n')

    def isdir(self, path):
        try:
            run_remote_command(f"stat '{path}' > /dev/null 2>&1 && [ -d '{path}' ] && echo 'True'")
            # stdoutから 'True' が返ることを期待するが、exec_commandの仕様上判定が難しい
            # 存在判定と分けるため、ここでは簡易的にtry-exceptで存在チェックのみとする
            return True # exists() がTrueなら存在すると仮定
        except Exception:
            return False

    def exists(self, path):
        try:
            # -e はファイルまたはディレクトリの存在チェック
            run_remote_command(f"ls '{path}' > /dev/null 2>&1")
            return True
        except Exception:
            return False

    def join(self, *args):
        # Linuxパス形式で結合
        return '/'.join(args).replace('//', '/')

    def makedirs(self, path, exist_ok=True):
        if exist_ok:
            run_remote_command(f"mkdir -p '{path}'")
        else:
            run_remote_command(f"mkdir '{path}'")

    def copy_file(self, src, dst):
        # リモート上でcpコマンドを実行
        run_remote_command(f"cp '{src}' '{dst}'")
    
    def rename(self, src, dst):
        # リモート上でmvコマンドを実行
        run_remote_command(f"mv '{src}' '{dst}'")

# --- JSON形式に対応するよう修正 ---
def get_shop_type(shop_code, shop_info):
    """店舗コードから新型/旧型を取得する (新しいJSON形式対応)"""
    shop_data = shop_info.get(str(shop_code)) # JSONのキーが文字列 "1289" のため変換
    if shop_data:
        return shop_data.get('型') # '型'キーから値を取得
    return None

def parse_filename(filename):
    """ファイル名から店舗コード、レーン番号を抽出する"""
    # 複雑なファイル名解析はローカルで実行
    parts = filename.split('_')
    if len(parts) >= 3:
        # [1.00]1304_011-01_2024-09-02...
        # 最初の部分を取得 e.g., "[1.00]1304" or "1304"
        first_part = parts[0]
        if ']' in first_part:
             shop_code_raw = first_part.split(']')[-1]
        else:
             shop_code_raw = first_part
             
        # shop_codeから不要な文字を削除し数値部分のみ抽出
        shop_code = ''.join(filter(str.isdigit, shop_code_raw))

        # レーン番号部分 e.g., "011-01"
        lane_part = parts[1]
        lane_code = lane_part.split('-')[-1] # e.g., "01"

        return shop_code, lane_code
    return None, None


def main():
    global ssh_client
    
    if USE_SSH:
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(SSH_HOSTNAME, SSH_PORT, SSH_USERNAME, SSH_PASSWORD)
        fm = SSHRemoteManager()
        print("SSH connection established and using remote file manager.")
    else:
        # USE_SSH=Falseの場合はこのスクリプトは意図した動作をしない
        # LocalFileManagerの実装は省略します。
        raise Exception("This script is configured for SSH only.")

    try:
        # 2. 保存先フォルダが存在する場合、リネーム処理の追加
        if fm.exists(DEST_BASE_PATH):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            new_dest_path = f"{DEST_BASE_PATH}_old_{timestamp}"
            print(f"Destination folder exists. Renaming {DEST_BASE_PATH} to {new_dest_path}")
            fm.rename(DEST_BASE_PATH, new_dest_path)
            # 新しい保存先を改めて作成
            fm.makedirs(DEST_BASE_PATH, exist_ok=True)
        else:
            fm.makedirs(DEST_BASE_PATH, exist_ok=True)
            
        # 3. ExcelファイルとJSONファイルの読み込み (これらはローカルで実行)
        df = pd.read_excel(EXCEL_PATH)
        # 結合セルに対応するため、'クラス名'と'クラス分類'列のNaN値をffillで埋める
        if 'クラス名' in df.columns:
            df['クラス名'] = df['クラス名'].ffill()
        if 'クラス分類' in df.columns:
            df['クラス分類'] = df['クラス分類'].ffill()

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
        summary_data = defaultdict(lambda: defaultdict(int)) 

        for shop_type in SHOP_TYPES:
            print(f"--- Processing {shop_type} shops ---")
            type_dest_path = fm.join(DEST_BASE_PATH, shop_type)
            fm.makedirs(type_dest_path, exist_ok=True)

            for class_name, prod_codes in class_map.items():
                print(f"  Processing class: {class_name}")
                class_dest_path = fm.join(type_dest_path, class_name)
                # 5. フォルダが存在しなければ作成
                fm.makedirs(class_dest_path, exist_ok=True)

                available_images = []
                for prod_code in prod_codes:
                    # TODO: ここは実際のソースフォルダ名パターンに合わせてください (例: f"{prod_code}")
                    source_folder_name = f"{prod_code}_XXXXX" 
                    source_folder_path = fm.join(SOURCE_BASE_PATH, source_folder_name)
                    
                    if fm.exists(source_folder_path):
                        # リモートでlsコマンドを実行し、ファイル名リストを取得
                        filenames = run_remote_command(f"ls '{source_folder_path}'").split('\n')
                        for filename in filenames:
                            if filename and (filename.lower().endswith(('.png', '.raw'))):
                                # ファイル名の解析はローカルPythonで行う
                                shop_code, lane_code = parse_filename(filename)
                                if shop_code and get_shop_type(shop_code, shop_info) == shop_type:
                                    available_images.append({
                                        'path': fm.join(source_folder_path, filename),
                                        'shop_code': shop_code,
                                        'lane_code': lane_code,
                                        'filename': filename
                                    })
                
                # ピックアップロジック (144枚) は前回と同じ
                target_count = 144
                selected_images = []
                used_combinations = set() 
                
                # 1回目のループで各店舗・レーン1枚ずつ取る
                for img in available_images:
                    combo = (img['shop_code'], img['lane_code'])
                    if combo not in used_combinations and len(selected_images) < target_count:
                        selected_images.append(img)
                        used_combinations.add(combo)
                        summary_data[class_name][f"{img['shop_code']}_L{img['lane_code']}"] += 1
                
                # 144枚に満たない場合、再ループして取る (無限ループにはならない)
                if len(selected_images) < target_count and len(available_images) > 0:
                    loops_needed = math.ceil((target_count - len(selected_images)) / len(available_images))
                    for _ in range(loops_needed):
                        for img in available_images:
                             if len(selected_images) < target_count:
                                selected_images.append(img)
                                summary_data[class_name][f"{img['shop_code']}_L{img['lane_code']}"] += 1
                
                # ファイルのコピー (リモートcpコマンドを使用)
                print(f"    Copying {len(selected_images)} images to {class_dest_path} using remote cp...")
                for img in selected_images:
                    dest_file_path = fm.join(class_dest_path, img['filename'])
                    # リモートでコピー先にファイルが存在しないことを確認してからコピー
                    if not fm.exists(dest_file_path):
                         fm.copy_file(img['path'], dest_file_path)
                print(f"    Finished class {class_name}.")

        # 7. エクセルに出力（CSVファイルとして出力）
        # このCSVファイルはWindowsローカルPCに出力します
        output_csv_path = 'image_composition_summary.csv' 
        print(f"Writing summary to local file {output_csv_path}")

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
        if ssh_client:
            ssh_client.close()
            print("SSH connection closed.")

if __name__ == "__main__":
    main()
import re # 新たにインポート

# ... (他のインポートは省略)

class SSHRemoteManager:
    """SSHコマンド実行ベースのファイル操作ラッパークラス"""
    # ... (listdir関数以外は省略)

    def listdir(self, path):
        # ls -F でディレクトリには / をつける
        output = run_remote_command(f"ls -F '{path}'")
        if not output:
            return []
        # ディレクトリ名末尾のスラッシュを削除して返すように修正
        return [f.rstrip('/') for f in output.split('\n')]
    
    # ... (isdir, exists, join, makedirs, copy_file, rename は省略)

# ... (get_shop_type関数は省略)

def parse_filename(filename):
    """ファイル名から店舗コード、レーン番号を抽出する"""
    # 複雑なファイル名解析はローカルで実行
    parts = filename.split('_')
    if len(parts) >= 3:
        # [1.00]1304_011-01_2024-09-02...
        # 最初の部分を取得 e.g., "[1.00]1304" or "1304"
        first_part = parts[0] # インデックス0を指定するよう修正
        if ']' in first_part:
             shop_code_raw = first_part.split(']')[-1]
        else:
             shop_code_raw = first_part
             
        # shop_codeから不要な文字を削除し数値部分のみ抽出
        shop_code = ''.join(filter(str.isdigit, shop_code_raw))

        # レーン番号部分 e.g., "011-01"
        lane_part = parts[1] # インデックス1を指定するよう修正
        lane_code = lane_part.split('-')[-1] # e.g., "01"

        return shop_code, lane_code
    return None, None


def main():
    # ... (SSH接続処理は省略)

    try:
        # ... (2. 保存先フォルダのリネーム処理は省略)
            
        # ... (3. Excel/JSONの読み込み、4. Excelデータの処理は省略)

        # 5 & 6. 画像のピックアップとコピー
        SHOP_TYPES = ['新型', '旧型']
        summary_data = defaultdict(lambda: defaultdict(int)) 

        # --- 追加箇所: リモートのソースディレクトリ直下の全フォルダ名を一度だけ取得 ---
        print(f"Listing all source folders in {SOURCE_BASE_PATH}...")
        all_source_folders = fm.listdir(SOURCE_BASE_PATH)
        print(f"Found {len(all_source_folders)} folders.")

        for shop_type in SHOP_TYPES:
            print(f"--- Processing {shop_type} shops ---")
            # ... (type_dest_pathの設定とmakedirsは省略)

            for class_name, prod_codes in class_map.items():
                print(f"  Processing class: {class_name}")
                # ... (class_dest_pathの設定とmakedirsは省略)

                available_images = []
                for prod_code in prod_codes:
                    # --- 修正箇所: prod_codeを含むフォルダを検索 ---
                    for folder_name in all_source_folders:
                        # 正規表現を使って、フォルダ名が 'prod_code_XXXX' または 'index_prod_code_XXXX' のパターンに一致するか確認
                        # prod_codeがフォルダ名の一部として含まれていればOKとする
                        if re.search(f"(^|_){re.escape(prod_code)}(_|$)", folder_name):
                            source_folder_path = fm.join(SOURCE_BASE_PATH, folder_name)
                            
                            # リモートでlsコマンドを実行し、ファイル名リストを取得
                            filenames = run_remote_command(f"ls '{source_folder_path}'").split('\n')
                            for filename in filenames:
                                if filename and (filename.lower().endswith(('.png', '.raw'))):
                                    shop_code, lane_code = parse_filename(filename)
                                    if shop_code and get_shop_type(shop_code, shop_info) == shop_type:
                                        available_images.append({
                                            'path': fm.join(source_folder_path, filename),
                                            'shop_code': shop_code,
                                            'lane_code': lane_code,
                                            'filename': filename
                                        })
                    
                # ... (ピックアップロジック (144枚) とファイルコピーロジックは省略)
                # ... (7. エクセルに出力（CSVファイルとして出力）ロジックは省略)

    # ... (エラーハンドリングとfinallyブロックは省略)

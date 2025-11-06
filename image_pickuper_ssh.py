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
import re
import random
from collections import defaultdict, deque

EXCEL_PATH = 'class_info.xlsx'
JSON_PATH = 'shopinfo.json'
SHEET_NAME = ''
SOURCE_BASE_PATH = '/path/to/remote/sdcard/images' 
DEST_BASE_PATH = '/path/to/remote/destination'
NEWTYPE_PATH = DEST_BASE_PATH + "/新型"
OLDTYPE_PATH = DEST_BASE_PATH + "/旧型"
IMAGES_PER_LANE = 2

USE_SSH = True 
SSH_HOSTNAME = 'your_remote_host'
SSH_PORT = 22
SSH_USERNAME = 'your_username'
SSH_PASSWORD = 'your_password'

ssh_client = None

def run_remote_command(command):
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

class LocalFileManager:
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

class SSHRemoteManager:
    # def listdir(self, path):
    #     # ls -F でディレクトリには / をつける
    #     output = run_remote_command(f"ls -F '{path}'")
    #     if not output:
    #         return []
    #     return output.split('\n')
    def listdir(self, path):
        output = run_remote_command(f"ls -F '{path}'")
        if not output:
            return []
        return [f.rstrip('/') for f in output.split('\n')]

    def isdir(self, path):
        try:
            run_remote_command(f"stat '{path}' > /dev/null 2>&1 && [ -d '{path}' ] && echo 'True'")
            return True
        except Exception:
            return False

    def exists(self, path):
        try:
            run_remote_command(f"ls '{path}' > /dev/null 2>&1")
            return True
        except Exception:
            return False

    def join(self, *args):
        return '/'.join(args).replace('//', '/')

    def makedirs(self, path, exist_ok=True):
        if exist_ok:
            run_remote_command(f"mkdir -p '{path}'")
        else:
            run_remote_command(f"mkdir '{path}'")

    def copy_file(self, src, dst):
        run_remote_command(f"cp '{src}' '{dst}'")
    
    def rename(self, src, dst):
        run_remote_command(f"mv '{src}' '{dst}'")

def get_shop_type(shop_code, shop_info):
    shop_data = shop_info.get(str(shop_code))
    if shop_data:
        return shop_data.get('型')
    return None

def parse_filename(filename):
    parts = filename.split('_')
    if len(parts) >= 3:
        first_part = parts[0]
        if ']' in first_part:
             shop_code_raw = first_part.split(']')[-1]
        else:
             shop_code_raw = first_part
        shop_code = ''.join(filter(str.isdigit, shop_code_raw))
        lane_part = parts[1]
        lane_code = lane_part.split('-')[-1]

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
        fm = LocalFileManager()
        print("Using local file manager.")

    try:
        if fm.exists(NEWTYPE_PATH):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            new_dest_path = f"{NEWTYPE_PATH}_old_{timestamp}"
            print(f"Destination folder exists. Renaming {NEWTYPE_PATH} to {new_dest_path}")
            fm.rename(NEWTYPE_PATH, new_dest_path)
            fm.makedirs(NEWTYPE_PATH, exist_ok=True)
        else:
            fm.makedirs(OLDTYPE_PATH, exist_ok=True)

        if fm.exists(OLDTYPE_PATH):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            new_dest_path = f"{OLDTYPE_PATH}_old_{timestamp}"
            print(f"Destination folder exists. Renaming {OLDTYPE_PATH} to {new_dest_path}")
            fm.rename(OLDTYPE_PATH, new_dest_path)
            fm.makedirs(OLDTYPE_PATH, exist_ok=True)
        else:
            fm.makedirs(OLDTYPE_PATH, exist_ok=True)
            
        df = pd.read_excel(EXCEL_PATH, SHEET_NAME)
        if 'クラス名' in df.columns:
            df['クラス名'] = df['クラス名'].ffill()
        if 'クラス分類' in df.columns:
            df['クラス分類'] = df['クラス分類'].ffill()

        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            shop_info = json.load(f)
        
        class_map = defaultdict(list)
        for index, row in df.iterrows():
            prod_code = str(row['商品コード']).strip()
            class_name = str(row['クラス名']).strip()
            if prod_code and class_name:
                class_map[class_name].append(prod_code)
        
        SHOP_TYPES = ['新型', '旧型']
        summary_data = defaultdict(lambda: defaultdict(int)) 

        print(f"Listing all source folders in {SOURCE_BASE_PATH}...")
        all_source_folders = fm.listdir(SOURCE_BASE_PATH)
        print(f"Found {len(all_source_folders)} folders.")

        for shop_type in SHOP_TYPES:
            print(f"--- Processing {shop_type} shops ---")
            type_dest_path = fm.join(DEST_BASE_PATH, shop_type)
            fm.makedirs(type_dest_path, exist_ok=True)

            for class_name, prod_codes in class_map.items():
                print(f"  Processing class: {class_name}")
                class_dest_path = fm.join(type_dest_path, class_name)
                fm.makedirs(class_dest_path, exist_ok=True)

                available_images = []
                
                for prod_code in prod_codes:
                    for folder_name in all_source_folders:
                        if re.search(f"(^|_){re.escape(prod_code)}(_|$)", folder_name):
                            source_folder_path = fm.join(SOURCE_BASE_PATH, folder_name)                  
                            try:
                                filenames = fm.listdir(source_folder_path)
                            except Exception:
                                continue
                            
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
                
                target_count = 144
                # images_per_lane = IMAGES_PER_LANE
                # selected_images = []
                # lane_counts = defaultdict(int) # (shop_code, lane_code) -> count

                if not available_images:
                    print("    No images available for this class.")
                    continue 

                buckets = defaultdict(list)
                for img in available_images:
                    key = (img['shop_code'], img['lane_code'])
                    buckets[key].append(img)
            
                for key in buckets:
                    random.shuffle(buckets[key])

                rr = [(key, deque(imgs)) for key, imgs in sorted(buckets.items())] 
                selected_images = []
                summary_counts = defaultdict(int) 

                while len(selected_images) < target_count and rr:
                    progressed = False
                    new_rr = []
                    for key, q in rr:
                        if len(selected_images) >= target_count:
                            break
                        if q:
                            img = q.popleft()
                            selected_images.append(img)
                            summary_counts[f"{key[0]}_{key[1]}"] += 1
                            progressed = True
                            if q:
                                new_rr.append((key, q))
                    rr = new_rr
                    if not progressed:
                        break
                        
                print(f"    Copying {len(selected_images)} images to {class_dest_path} using remote cp...")
                for img in selected_images:
                    dest_file_path = fm.join(class_dest_path, img['filename'])
                    if not fm.exists(dest_file_path):
                         fm.copy_file(img['path'], dest_file_path)
                print(f"    Finished class {class_name}.")
                
            for combo, cnt in summary_counts.items():
                summary_data[class_name][combo] += cnt
        
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

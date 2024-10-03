import os
import random
import shutil


def balance_num(num_files, target_files, folder_path, files):
    if num_files > target_files:
        num_files_to_remove = num_files - target_files
        print(f"\n文件数量超过 {target_files}，将删除多余的 {num_files_to_remove} 个文件...")

        # 删除超过数量的文件
        for i in range(num_files_to_remove):
            file_to_remove = os.path.join(folder_path, files[i])
            os.remove(file_to_remove)
            print(f"已删除文件: {file_to_remove}")

        print(f"\n删除完成，文件夹 {folder_path} 中的文件数量现在为 {target_files}")

    else:
        num_files_to_copy = target_files - num_files
        print(f"\n文件数量少于 {target_files}，将复制 {num_files_to_copy} 个文件...")

        # 获取可以复制的文件列表
        files_to_copy = random.sample(files, num_files_to_copy)

        # 复制文件
        for file_to_copy in files_to_copy:
            source_path = os.path.join(folder_path, file_to_copy)
            destination_path = os.path.join(folder_path, f"copy_{file_to_copy}")
            shutil.copy2(source_path, destination_path)
            print(f"已复制文件: {file_to_copy}")

        print(f"\n复制完成，文件夹 {folder_path} 中的文件数量现在为 {target_files}")


def count_files_in_folder(folder_path, target_files=6000):
    try:
        # 获取文件夹中的所有文件
        files = os.listdir(folder_path)

        # 计算文件数量
        num_files = len(files)
        print(f"\n文件夹 {folder_path} 中的文件数量: {num_files}")

        # 如果文件数量超过设定的最大值，删除多余的文件
        balance_num(num_files, target_files, folder_path, files)    # 删除多于操作

    except FileNotFoundError:
        print(f"找不到文件夹: {folder_path}")
    except Exception as e:
        print(f"发生错误: {e}")


# 硬编码文件夹路径
folder_path = "us06/sti"

# 调用函数检查文件夹中的文件数量
count_files_in_folder(folder_path)

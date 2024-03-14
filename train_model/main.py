import os

def list_pictures_in_folders(folder_path):
    folders_list = [item for item in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, item))]
    pictures_in_folders = []

    for folder in folders_list:
        folder_path_full = os.path.join(folder_path, folder)
        all_files = os.listdir(folder_path_full)
        pictures = [file for file in all_files if file.endswith(('.jpg', '.png', '.jpeg'))]  # Add other picture formats if needed
        pictures_in_folders.append({folder: pictures})

    return pictures_in_folders

# Replace 'your_folder_path_here' with the actual path of your folder
folder_path = 'train_bearing_image/'
pictures_list = list_pictures_in_folders(folder_path)
print(pictures_list)
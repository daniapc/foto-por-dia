import os

folder = 'selfies'  # pasta onde estão as imagens

for filename in os.listdir(folder):
    if filename.startswith('IMG_'):
        old_path = os.path.join(folder, filename)
        new_filename = filename.replace('IMG_', '', 1)
        new_path = os.path.join(folder, new_filename)

        os.rename(old_path, new_path)
        print(f'Renomeado: {filename} → {new_filename}')

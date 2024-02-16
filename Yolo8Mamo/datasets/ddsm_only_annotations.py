# en este script filtramos el dataset de DDSM para que solo contenga las imagenes que tienen anotaciones

import pathlib
import os


def get_number_of_annotations(label_path):
    """
    Get the number of annotations in a label file
    """
    
    with open(label_path, 'r') as f:
        return len(f.readlines())
    


def filter_dataset(original_dataset_path, dest_dataset_path):
    """
    Filter the dataset to only contain images that have annotations
    """
    
    original_dataset_path = pathlib.Path(original_dataset_path)
    dest_dataset_path = pathlib.Path(dest_dataset_path)
    
    for partition in ['training', 'validation']:
        orig_images_folder = original_dataset_path / partition / 'images'
        orig_labels_folder = original_dataset_path / partition / 'labels'
        dest_images_folder = dest_dataset_path / partition / 'images'
        dest_labels_folder = dest_dataset_path / partition / 'labels'
        
        os.makedirs(dest_images_folder, exist_ok=True)
        os.makedirs(dest_labels_folder, exist_ok=True)
        num_discarded = 0
        num_selected = 0
        
        print(f"Processing {partition} partition: {orig_images_folder} -> {dest_images_folder}")
        for k, image_path in enumerate(orig_images_folder.glob('*.png')):
            label_path = orig_labels_folder / (image_path.stem + '.txt')
            if label_path.exists() and get_number_of_annotations(label_path) > 0:
                dest_image_path = dest_images_folder / image_path.name
                dest_label_path = dest_labels_folder / label_path.name
                os.link(image_path, dest_image_path)
                os.link(label_path, dest_label_path)
                num_selected += 1
            else:
                num_discarded += 1
        print("Finished partition", partition)
        print(f"Images processed: {k+1}")
        print(f"Discarded {num_discarded} images from {partition} partition")
        print(f"Selected {num_selected} images from {partition} partition")
        
        








if __name__ == "__main__":
    original_dataset_path ='/home/alalbiol/Data/mamo/ddsm_yolo'
    dest_dataset = '/home/alalbiol/Data/mamo/ddsm_yolo_only_annotations'
    
    filter_dataset(original_dataset_path, dest_dataset)
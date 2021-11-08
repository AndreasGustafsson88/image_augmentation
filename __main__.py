from image_augmenter.models.data_augmentation import PolyAugmenter
from configparser import ConfigParser


if __name__ == '__main__':
    config = ConfigParser()
    config.read('settings.ini')
    config.get('paths', 'ANNOTATIONS_FILE')

    my_augmentation = PolyAugmenter(config.get('paths', 'ANNOTATIONS_FILE'),
                                    config.get('paths', 'IMAGE_PATH'),
                                    validate=False)

    my_augmentation.augment_images()
    my_augmentation.save_all(config.get('paths', 'OUTPUT_DIR'))

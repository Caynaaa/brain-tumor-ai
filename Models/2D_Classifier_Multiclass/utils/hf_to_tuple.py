def hf_dataset_to_tuple(hf_dataset, image_key='image', label_key='label'):
    """
    Converts a Hugging Face dataset to a tuple of (images, labels).

    Args:
        hf_dataset (Dataset): The Hugging Face dataset to convert.
        image_key (str): The key for the image data in the dataset. Defaults to 'image'.
        label_key (str): The key for the label data in the dataset. Defaults to 'label'.

    Returns:
        tuple: A tuple containing two lists:
            - List of images.
            - List of labels.
    """
    images = [item[image_key] for item in hf_dataset]
    labels = [item[label_key] for item in hf_dataset]
    return images, labels
    
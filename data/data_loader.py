# encoding =utf-8

def CreateDataLoader(opt):
    """
    :param opt:
    :return:
    """
    from data.custom_dataset_data_loader import CustomDatasetDataLoader

    data_loader = CustomDatasetDataLoader()
    print('DataLoader: ', data_loader.name())

    data_loader.initialize(opt)

    return data_loader

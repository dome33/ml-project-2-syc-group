from mltu.torch.dataProvider import DataProvider


def split_data_provider(dataprovider,split=0.9): 
    split = int(len(dataprovider._dataset) * split) 
    train_dataProvider = DataProvider(
        dataset=dataprovider._dataset[:split],
        batch_size=dataprovider._batch_size,
        data_preprocessors=dataprovider._data_preprocessors,
        transformers=dataprovider._transformers,
        use_cache=dataprovider._use_cache,
    )

    test_dataProvider = DataProvider(
        dataset=dataprovider._dataset[split:],
        batch_size=dataprovider._batch_size,
        data_preprocessors=dataprovider._data_preprocessors,
        transformers=dataprovider._transformers,
        use_cache=dataprovider._use_cache,
    )
    return train_dataProvider, test_dataProvider 

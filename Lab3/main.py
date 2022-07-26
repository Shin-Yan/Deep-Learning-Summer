from dataloader import RetinopathyLoader
test_dataset = RetinopathyLoader('./data', 'test')
test_dataset[50]
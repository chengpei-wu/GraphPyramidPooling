pooling_sizes = [1, 2, 4, 8, 15]

epochs = 200
batch_size = 4
valid = 0.1

training_size = '(700,1300)'
testing_size = training_size
isd = 0
isw = 0
robustness = 'yc'
if robustness == 'yc':
    atk = 'nrnd'
else:
    atk = 'ndeg'

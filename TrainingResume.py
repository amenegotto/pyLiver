# PURPOSE:
# create a structure for resume training after session is lost

from ExecutionAttributes import ExecutionAttribute

# Default serialization is slow to write and read from disk
def save_execution_attributes(attr : ExecutionAttribute, path):
    print("Saving execution attributes to ", path) 
    with open(path, 'w') as f:
        f.write('seq='+str(attr.seq)+ '\n')
        f.write('img_width='+str(attr.img_width)+ '\n')
        f.write('img_height='+str(attr.img_height)+ '\n')
        f.write('path='+attr.path+ '\n')
        f.write('summ_basename='+attr.summ_basename+ '\n')
        f.write('epochs='+str(attr.epochs)+ '\n')
        f.write('batch_size='+str(attr.batch_size)+ '\n')
        f.write('train_data_dir='+attr.train_data_dir+ '\n')
        f.write('validation_data_dir='+attr.validation_data_dir+ '\n')
        f.write('test_data_dir='+attr.test_data_dir+ '\n')
        f.write('steps_train='+str(attr.steps_train)+ '\n')
        f.write('steps_valid='+str(attr.steps_valid)+ '\n')
        f.write('steps_test='+str(attr.steps_test)+ '\n')
        f.write('architecture='+attr.architecture+ '\n')
        f.write('curr_basename='+attr.curr_basename+ '\n')
        f.close()

def read_attributes(path):
    print("Reading execution attributes from ", path) 
    attr = ExecutionAttribute()

    lines = [line.rstrip('\n') for line in open(path)]

    attr.seq = int(lines[0].split("=",1)[1])
    attr.img_width = int(lines[1].split("=",1)[1])
    attr.img_height = int(lines[2].split("=",1)[1])
    attr.path = lines[3].split("=",1)[1]
    attr.summ_basename = lines[4].split("=",1)[1]
    attr.epochs = int(lines[5].split("=",1)[1])
    attr.batch_size = int(lines[6].split("=",1)[1])
    attr.train_data_dir = lines[7].split("=",1)[1]
    attr.validation_data_dir = lines[8].split("=",1)[1]
    attr.test_data_dir = lines[9].split("=",1)[1]
    attr.steps_train = int(lines[10].split("=",1)[1])
    attr.steps_valid = int(lines[11].split("=",1)[1])
    attr.steps_test = int(lines[12].split("=",1)[1])
    attr.architecture = lines[13].split("=",1)[1]
    attr.curr_basename = lines[14].split("=",1)[1]

    return attr

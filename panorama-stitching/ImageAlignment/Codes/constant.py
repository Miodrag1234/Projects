# training dataset path – UDIS-D training (input1/input2)
TRAIN_FOLDER = '../training'

# testing dataset path – UDIS-D testing (input1/input2)
TEST_FOLDER = '../testing'

# GPU index ('0' za prvu karticu)
GPU = '0'

#batch size for training
TRAIN_BATCH_SIZE = 4

#batch size for testing
TEST_BATCH_SIZE = 1

# broj iteracija – start na 1M (od originalnog ckpt), dodajemo 200k samo na Net4
ITERATIONS = 1200000

# checkpoints path – nova mapa za Net4-only fine-tune (start: original 1M)
SNAPSHOT_DIR = "./checkpoints_homo3"

#sumary path
SUMMARY_DIR = "./summary"



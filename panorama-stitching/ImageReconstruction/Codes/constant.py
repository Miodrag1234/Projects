#training dataset path (samo za trening, za evaluaciju nije potrebno)
TRAIN_FOLDER = '../../ImageAlignment/output/training'

# testing dataset path – izlaz Stage 1 (ImageAlignment): warp1, warp2, mask1, mask2
# Tipično: ImageAlignment/output/testing (nakon output_inference.py)
TEST_FOLDER = '../../ImageAlignment/output/testing'

# mapa za spremanje šivanih slika (evaluacija)
RESULTS_DIR = '../results2'

# GPU index (npr. '0' za jednu karticu)
GPU = '0'

#batch size for training
TRAIN_BATCH_SIZE = 1

#batch size for testing
TEST_BATCH_SIZE = 1

#num of iters
ITERATIONS = 200000

# checkpoints path – nova mapa za trening od nule, stari ostaju netaknuti
SNAPSHOT_DIR = "./checkpoints_new"

#sumary path
SUMMARY_DIR = "./summary"

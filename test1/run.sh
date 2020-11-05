batch_size="20"
kernel_sizes="3,4,5,6,7"
dim="200"
kernel_num="10"
learning_rate="0.01"
epochs="10"
batch_num="200"


python main.py \
--batch-size=${batch_size} \
--kernel-sizes=${kernel_sizes} \
--dim=${dim} \
--kernel-num=${kernel_num} \
--lr=${learning_rate} \
--epochs=${epochs} \
--batch-num=${batch_num} \
--shuffle --train

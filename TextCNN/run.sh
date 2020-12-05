batch_size="50"
kernel_sizes="3,4,5"
dim="300"
kernel_num="50"
learning_rate="0.001"
epochs="100"
batch_num="10000"



python main.py \
--batch-size=${batch_size} \
--kernel-sizes=${kernel_sizes} \
--dim=${dim} \
--kernel-num=${kernel_num} \
--lr=${learning_rate} \
--epochs=${epochs} \
--batch-num=${batch_num} \
--shuffle --train
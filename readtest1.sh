./drop_caches.sh
md5sum /mnt/isilon1/data/imagenet-scratch/tfrecords/* &
md5sum /mnt/isilon2/data/imagenet-scratch/tfrecords/* &
md5sum /mnt/isilon3/data/imagenet-scratch/tfrecords/* &
md5sum /mnt/isilon4/data/imagenet-scratch/tfrecords/* &
wait



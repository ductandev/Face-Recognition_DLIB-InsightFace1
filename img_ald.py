from preprocesss import preprocessess

input_datadir = './Output'
output_datadir = './datasets/train'

obj=preprocessess(input_datadir,output_datadir)
nrof_images_total , nrof_successfully_aligned=obj.collect_data()

print('Total number of images: %d' % nrof_images_total)
print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
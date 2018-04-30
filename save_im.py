import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

model_dir = "./model/"
model_name = "gan_mnist"

def sample_z( size ):
    return np.random.uniform(-1. , 1. , size=size )

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


sess = tf.Session()

saver = tf.train.import_meta_graph( model_dir + model_name + ".meta" )
saver.restore(sess, tf.train.latest_checkpoint(model_dir))

graph = tf.get_default_graph()

# restore placeholder

Z = graph.get_tensor_by_name("Placeholder_1:0")

# restore generator output op
gene_out = graph.get_tensor_by_name("generator/Sigmoid:0")

batch_size = 5
z_dim = 100

gen_output = sess.run(gene_out, feed_dict={Z : sample_z([batch_size, z_dim])})
fig = plot(gen_output)

plt.savefig('out_sample.png', bbox_inches='tight')
plt.close(fig)


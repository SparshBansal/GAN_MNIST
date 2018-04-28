import tensorflow as tf 
from PIL import Image
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def generator(z):

    input_dims = z.shape[1]
    output_dims = 128
    
    old_vars = tf.global_variables()
    
    print [input_dims, output_dims]

    w1 = tf.get_variable('weight_l1' , shape=[input_dims, output_dims], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable('bias_l1' , shape=[output_dims], dtype=tf.float32, initializer=tf.zeros_initializer())

    w2 = tf.get_variable('weight_l2', shape=[128, 784], dtype=tf.float32 , initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable('bias_l2', shape=[784], dtype=tf.float32, initializer=tf.zeros_initializer())
    
    o1 = tf.nn.relu( tf.matmul(z,w1) + b1 )
    o2 = tf.nn.sigmoid( tf.matmul(o1, w2) + b2 )
        
    new_vars = tf.global_variables()

    gen_vars = list(set(new_vars)-set(old_vars))
    return o2, gen_vars


def discriminator(x):
    
    input_dims = x.shape[1]
    output_dims = 128

    old_vars = tf.global_variables()

    w1 = tf.get_variable('weight_l1', shape=[input_dims, output_dims], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable('bias_l1', shape=[output_dims], dtype=tf.float32, initializer=tf.zeros_initializer())

    w2 = tf.get_variable('weight_l2' , shape=[128, 1], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable('bias_l2', shape=[1], dtype=tf.float32, initializer=tf.zeros_initializer())

    o1 = tf.nn.relu(tf.matmul(x,w1) + b1)
    z1 = tf.matmul(o1,w2) + b2
    o2 = tf.nn.sigmoid(z1)

    new_vars = tf.global_variables()
    disc_vars = list(set(new_vars) - set(old_vars))

    return o2, z1, disc_vars

def generator_loss( disc_logits ):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_logits, labels=tf.ones_like( disc_logits )))

def discriminator_loss( disc_logits_real, disc_logits_fake ):
    d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits( logits=disc_logits_real, labels=tf.ones_like(disc_logits_real))
    d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits( logits=disc_logits_fake, labels=tf.zeros_like(disc_logits_fake))
    
    d_loss = tf.reduce_mean(d_loss_real + d_loss_fake)
    return d_loss

def sample_z( size ):
    return np.random.uniform(-1. , 1. , size=size )

# input images for discriminator
X = tf.placeholder(tf.float32, shape=[None,784])

# latent vector for generator 
Z = tf.placeholder(tf.float32, shape=[None,100])

with tf.variable_scope('generator') as scope:
    gene_out, gene_vars = generator(Z)

with tf.variable_scope('discriminator') as scope:
    disc_out_real , disc_logits_real, disc_vars = discriminator(X)
    scope.reuse_variables()
    disc_out_fake , disc_logits_fake, _  = discriminator(gene_out)

with tf.name_scope('generator_loss'):
    g_loss = generator_loss( disc_logits_fake )

with tf.name_scope('discriminator_loss'):
    d_loss = discriminator_loss( disc_logits_real , disc_logits_fake )

g_optimizer = tf.train.AdamOptimizer()
g_train_step = g_optimizer.minimize(g_loss, var_list=gene_vars)

d_opitmizer = tf.train.AdamOptimizer()
d_train_step = d_opitmizer.minimize(d_loss, var_list=disc_vars)

batch_size=128

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer= tf.summary.FileWriter('./graph/')
    writer.add_graph(sess.graph)
    
    # now train generator and discriminator alternatively 
    for it in range(1000000):
        X_batch , _ = mnist.train.next_batch(batch_size)
        _ , _ , g_loss_current, d_loss_current = sess.run([d_train_step, g_train_step, g_loss, d_loss] , feed_dict={X: X_batch, Z:sample_z([batch_size, 100])})

        if it % 1000 == 0:
            print('Iter: {}'.format(it))
            print('D loss: {:.4}'. format(d_loss_current))
            print('G_loss: {:.4}'.format(g_loss_current))
            print()

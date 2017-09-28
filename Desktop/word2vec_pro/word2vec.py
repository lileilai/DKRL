import collections
import os
import random
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import math
# Read the data into a list of strings.
def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words"""
  data=open("./description.txt")
  word_s=[]
  for each_line in data:
    for w in each_line.split(" "):
      word_s.append(w)
  return word_s

words = read_data("./description.txt")
print('Data size', len(words))
# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 50000

def build_dataset(words):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
del words  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1 # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [ skip_window ]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels

batch, labels = generate_batch(batch_size=12, num_skips=6, skip_window=3)
for i in range(12):
  print(batch[i], reverse_dictionary[batch[i]],
      '->', labels[i, 0], reverse_dictionary[labels[i, 0]])
# Step 4: Build and train a skip-gram model.

unigrams = [ c / vocabulary_size for token, c in count ]
batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 2       # How many words to consider left and right.
num_skips = 1         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    # embeddings = tf.Variable(
    #     tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    # embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    #
    # # Construct the variables for the NCE loss
    # nce_weights = tf.Variable(
    #     tf.truncated_normal([vocabulary_size, embedding_size],
    #                         stddev=1.0 / math.sqrt(embedding_size)))
    # nce_biases = tf.Variable(tf.zeros([vocabulary_size]))


    input_ids = train_inputs
    labels = tf.reshape(train_labels, [batch_size])
    # [vocabulary_size, emb_dim] - input vectors
    input_vectors = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0),
        name="input_vectors")

    # [vocabulary_size, emb_dim] - output vectors
    output_vectors = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0),
        name="output_vectors")

    # [batch_size, 1] - labels
    labels_matrix = tf.reshape(
        tf.cast(labels,
                dtype=tf.int64),
        [batch_size, 1])

    # Negative sampling.
    sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
        true_classes=labels_matrix,
        num_true=1,
        num_sampled=200,
        unique=True,
        range_max=vocabulary_size,
        distortion=0.75,
        unigrams=unigrams))

    # [batch_size, emb_dim] - Input vectors for center words
    center_vects = tf.nn.embedding_lookup(input_vectors, input_ids)
    # [batch_size, emb_dim] - Output vectors for context words that
    # (center_word, context_word) is in corpus
    context_vects = tf.nn.embedding_lookup(output_vectors, labels)
    # [num_sampled, emb_dim] - vector for sampled words that
    # (center_word, sampled_word) probably isn't in corpus
    sampled_vects = tf.nn.embedding_lookup(output_vectors, sampled_ids)
    # compute logits for pairs of words that are in corpus
    # [batch_size, 1]
    incorpus_logits = tf.reduce_sum(tf.mul(center_vects, context_vects), 1)
    incorpus_probabilities = tf.nn.sigmoid(incorpus_logits)

    # Sampled logits: [batch_size, num_sampled]
    # We replicate sampled noise labels for all examples in the batch
    # using the matmul.
    sampled_logits = tf.matmul(center_vects,
                               sampled_vects,
                               transpose_b=True)
    outcorpus_probabilities = tf.nn.sigmoid(-sampled_logits)

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  # [batch_size, 1]
  outcorpus_loss_perexample = tf.reduce_sum(tf.log(outcorpus_probabilities), 1)
  loss_perexample = - tf.log(incorpus_probabilities) - outcorpus_loss_perexample

  loss =  tf.reduce_sum(loss_perexample) / batch_size

  # Construct the SGD optimizer using a learning rate of 0.4.
  optimizer = tf.train.GradientDescentOptimizer(.4).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(input_vectors + output_vectors), 1, keep_dims=True))
  normalized_embeddings = (input_vectors + output_vectors) / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  # Add variable initializer.
  init = tf.initialize_all_variables()

# Step 5: Begin training.
num_steps = 200000
# num_steps = 100001

with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  init.run()
  print("Initialized")
  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 1000 == 0:
      if step > 0:
        average_loss /= 1000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print("Average loss at step ", step, ": ", average_loss)
      average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8 # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k+1]
        log_str = "Nearest to %s:" % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = "%s %s," % (log_str, close_word)
        print(log_str)
  final_embeddings = normalized_embeddings.eval()
  f=open("./word_vec.txt","w+")
  for each_line in final_embeddings:
    print each_line
    f.write(each_line+"\n")

# Step 6: Visualize the embeddings.

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(18, 18))  #in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i,:]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

try:
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels)

except ImportError:
  print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")

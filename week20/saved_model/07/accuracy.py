import tensorflow as tf

# l1 4, 2, 3
l1 = [[[1, 0, 0], [0, 1, 0]],
      [[0, 1, 0], [0, 1, 0]],
      [[0, 0, 1], [0, 1, 0]],
      [[1, 0, 0], [0, 1, 0]]
      ]

# l2 4, 2, 3
l2 = [[[0, 13, 0], [0, 1, 0]],
      [[0, 1, 0], [0, 1, 0]],
      [[0, 0, 1], [0, 1, 0]],
      [[1, 0, 0], [0, 1, 0]]
      ]

logits = tf.placeholder(tf.int64, [None, 2, 3])
labels = tf.placeholder(tf.int64, [None, 2, 3])

acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(logits, 2), predictions=tf.argmax(labels, 2))

sess = tf.Session()

sess.run(tf.local_variables_initializer())
sess.run(tf.global_variables_initializer())

print('acc:', sess.run(acc_op, {logits: l1, labels: l2}))

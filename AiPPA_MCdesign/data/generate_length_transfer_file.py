import pickle

# length distribution
# the sum should equal to 1
a = [0.3934, 0.6066]   # cdr2: 5-6
b = [0.0146, 0.0325, 0.052, 0.0478, 0.0681,
     0.0633, 0.0523, 0.0839, 0.0712, 0.0806,
     0.1205, 0.088, 0.0827, 0.069, 0.0355,
     0.038]    # cdr3: 5-20
cdr2 = {5: 0.3934, 6: 0.6066}
cdr3 = {5: 0.0146, 6: 0.0325, 7: 0.052, 8: 0.0478, 9: 0.0681,
        10: 0.0633, 11: 0.0523, 12: 0.0839, 13: 0.0712,
        14: 0.0806, 15: 0.1205, 16: 0.088, 17: 0.0827,
        18: 0.069, 19: 0.0355, 20: 0.038}

print(sum(a))
print(sum(b))
# print(dict(zip(range(5, 7), a)))
# print(dict(zip(range(5, 21), b)))

cdr3_transfer_pro = {}
sum_tmp = 0
mut_prob = 0.70
for i in range(5, 21):
    idx = i - 5
    delete_prob = round(sum_tmp / (1 - b[idx]) * (1 - mut_prob), 5)
    insert_prob = round((1 - mut_prob) - delete_prob, 5)
    cdr3_transfer_pro[i] = {'mut': mut_prob,
                            'delete': delete_prob,
                            'insert': insert_prob}
    sum_tmp += b[idx]
print(cdr3_transfer_pro)

cdr2_transfer_pro = {5: {'mut': mut_prob,
                         'delete': 0,
                         'insert': round(1 - mut_prob, 2)},
                     6: {'mut': mut_prob,
                         'delete': round(1 - mut_prob, 2),
                         'insert': 0}}
print(cdr2_transfer_pro)

with open('cdr_length_transfer_prob.pkl', 'wb') as f:
    pickle.dump({'cdr2': cdr2_transfer_pro,
                 'cdr3': cdr3_transfer_pro},
                f)


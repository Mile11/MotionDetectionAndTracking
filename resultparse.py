file = open('results.txt')
lines = file.readlines()

avg_acc = [
    0, 0
]

avg_rel_acc = [
    0, 0
]

nums = [
    0, 0
]

i = 0

for l in lines:

    if l == '====================\n':
        i += 1
        continue

    split = l.split(' ')

    if len(split) != 8:
        continue

    nums[i] += 1
    avg_acc[i] += float(split[4])
    avg_rel_acc[i] += float(split[7])


avg_acc[0] /= nums[0]
avg_rel_acc[0] /= nums[0]
avg_acc[1] /= nums[1]
avg_rel_acc[1] /= nums[1]

print(avg_acc)
print(avg_rel_acc)
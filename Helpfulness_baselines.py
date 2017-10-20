import gzip
from collections import defaultdict

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)

### Rating baseline: compute averages for each user, or return the global average if we've never seen the user before

allRatings = []
userRatings = defaultdict(list)
for l in readGz("train.json.gz"):
  user,item = l['reviewerID'],l['itemID']
  allRatings.append(l['rating'])
  userRatings[user].append(l['rating'])

globalAverage = sum(allRatings) / len(allRatings)
userAverage = {}
for u in userRatings:
  userAverage[u] = sum(userRatings[u]) / len(userRatings[u])

predictions = open("predictions_Rating.txt", 'w')
for l in open("pairs_Rating.txt"):
  if l.startswith("userID"):
    #header
    predictions.write(l)
    continue
  u,i = l.strip().split('-')
  if u in userAverage:
    predictions.write(u + '-' + i + ',' + str(userAverage[u]) + '\n')
  else:
    predictions.write(u + '-' + i + ',' + str(globalAverage) + '\n')

predictions.close()

### Helpfulness baseline: similar to the above. Compute the global average helpfulness rate, and the average helpfulness rate for each user

allHelpful = []
userHelpful = defaultdict(list)

for l in readGz("train.json.gz"):
  user,item = l['reviewerID'],l['itemID']
  allHelpful.append(l['helpful'])
  userHelpful[user].append(l['helpful'])

averageRate = sum([x['nHelpful'] for x in allHelpful]) * 1.0 / sum([x['outOf'] for x in allHelpful])
userRate = {}
for u in userHelpful:
  totalU = sum([x['outOf'] for x in userHelpful[u]])
  if totalU > 0:
    userRate[u] = sum([x['nHelpful'] for x in userHelpful[u]]) * 1.0 / totalU
  else:
    userRate[u] = averageRate

predictions = open("predictions_Helpful.txt", 'w')
for l in open("pairs_Helpful.txt"):
  if l.startswith("userID"):
    #header
    predictions.write(l)
    continue
  u,i,outOf = l.strip().split('-')
  outOf = int(outOf)
  if u in userRate:
    predictions.write(u + '-' + i + '-' + str(outOf) + ',' + str(outOf*userRate[u]) + '\n')
  else:
    predictions.write(u + '-' + i + '-' + str(outOf) + ',' + str(outOf*averageRate) + '\n')

predictions.close()

### Category prediction baseline: Just consider some of the most common words from each category

catDict = {'Women': 0, 'Men': 1, 'Girls': 2, 'Boys': 3, 'Baby': 4}

predictions = open("predictions_Category.txt", 'w')
predictions.write("userID-reviewHash,category\n")
for l in readGz("test_Category.json.gz"):
  cat = catDict['Women'] # If there's no evidence, just choose the most common category in the dataset
  words = l['reviewText'].lower().split()
  if 'men' in words or 'man' in words:
    cat = catDict['Men']
  if 'girls' in words or 'girl' in words:
    cat = catDict['Girls']
  if 'boys' in words or 'boy' in words:
    cat = catDict['Boys']
  if 'babies' in words or 'baby' in words:
    cat = catDict['Baby']
  predictions.write(l['reviewerID'] + '-' + l['reviewHash'] + "," + str(cat) + "\n")

predictions.close()

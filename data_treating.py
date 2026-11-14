import json

text = open("fypurDeskGeneralChat.json", "r", encoding="utf-8").read()
data = json.loads(text)
messages = data["messages"]
print(len(data["messages"]))

# objective : turn this into something similar to tiny shakespeare

blocks = [
    f"""{m["author"]["nickname"]}
{m["content"]}"""
    for m in messages
]

out = "\n\n".join(blocks)
chars = sorted(list(set(out)))

limit = 100
for c in chars:
    count = out.count(c)
    if count < limit:
        out = out.replace(c, "")

chars = sorted(list(set(out)))
# print(chars)
# print(len(chars))

with open("fypurdeskchat.txt", "w", encoding="utf-8") as f:
    f.write(out)
import os
sum = 0
for i in range(100):
    cmd = "wafamole evade --model-type PyTorch wafamole/models/custom/example_models/ModelWAF3.pt \"admin' OR 1=1#\" | sed -n 's/.*and mutation Rounds = //p' >> Model3Input1.txt"
    print(cmd)
    os.system(cmd)
    # print("Hi", a)
    # sum = sum + int(a)
# print (sum)
# print(sum/3)

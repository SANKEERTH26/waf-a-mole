k = open("Model3Input1.txt", 'r')
lines = k.readlines()  # i am reading lines here
counter = 0  # counter update each time number is entered
for line in lines:  # taking each line
    conv_int = int(line)  # converting string to int
    counter = counter + conv_int  # update counter
print(counter/len(lines))
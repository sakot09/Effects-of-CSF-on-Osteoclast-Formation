import numpy as np

f4csf = np.loadtxt("F4wCSF.csv", delimiter=",")
f4none = np.loadtxt("F4noCSF.csv", delimiter=",")

f5csf = np.loadtxt("F5wCSF.csv", delimiter=",")
f5none = np.loadtxt("F5noCSF.csv", delimiter=",")

csf_totals = f4csf[:, 1] * f5csf[:, 1]

absence_totals = f4none[:,1] * f5none[:, 1]

x = [0,4,8,16,24,48,72]

csf = np.column_stack((x, csf_totals))
absent = np.column_stack((x, absence_totals))

np.savetxt("csfdata.csv", csf)
np.savetxt("csfabsent.csv", absent)

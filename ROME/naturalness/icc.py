import pingouin as pg
import pandas as pd
import pickle as pk
import os

def create_list(dic):
	#create DataFrame
	df = pd.DataFrame({'picture_idx': [i for i in range(1,501)]*3,
					'rater': ['A']*500+['B']*500+['C']*500 ,
					'rating': [score[0] for score in dic]+[score[1] for score in dic]+[score[2] for score in dic] })
	return df

if __name__ == "__main__":
	total = 0
	names = os.listdir("list_for_icc")
	i = 0
	for name in names:
		with open(os.path.join('list_for_icc',name), 'rb') as f:
			dic = pk.load(f)
		df = create_list(dic)
		icc = pg.intraclass_corr(data=df, targets='picture_idx', raters='rater', ratings='rating')
		# print('idx: ',i)
		# print(icc.set_index('Type'))
		# print(type(icc.set_index('Type')))
		total += icc.set_index('Type').lookup(('ICC2k',),('ICC',))
		i += 1
		print('ICC2K,pkl',i, icc.set_index('Type').lookup(('ICC2k',),('ICC',)))
	
	avg = total/i
	print("final_result",avg)
	# for i in range(1,9):
	# 	with open('session_'+str(i)+'_kappa.pkl','rb') as f:
	# 		dic = pk.load(f)
	# 	df = create_list(dic)
	# 	icc = pg.intraclass_corr(data=df, targets='picture_idx', raters='rater', ratings='rating')
	# 	print('idx: ',i)
	# 	print(icc.set_index('Type'))
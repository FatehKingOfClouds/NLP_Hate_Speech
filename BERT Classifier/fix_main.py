import re

with open('BERT_training_inference.py', 'r') as f:
    content = f.read()

# Find the if __name__ block and replace the entire nested loop structure
pattern = r"if __name__=='__main__':[^}]*"

replacement = """if __name__=='__main__':

	# SIMPLE MODE: Run with params defined above
	best_val_fscore = 0
	_,best_val_fscore = train_model(params, best_val_fscore)
	print('============================')
	print(f'Model for Language {params["language"]} trained successfully!')
	print('============================' )
"""

# Use a more targeted approach - replace from if __name__ to end of file
start_idx = content.find("if __name__=='__main__':")
if start_idx != -1:
    content = content[:start_idx] + replacement

with open('BERT_training_inference.py', 'w') as f:
    f.write(content)

print("File updated successfully!")

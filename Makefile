move_code_to_server:
	rsync -avz --exclude-from="exclude_me.txt" ./ shamil@10.241.1.217:/home/shamil/thesis_sketch/



lint:
	flake8 . -j 4  --max-line-length 100 --exit-zero
	isort . -j 4



run:
	cd src/train_bm25; PYTHONPATH=../.. python train_collie.py
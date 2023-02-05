move_to_server:
	rsync -avz ./ shamil@172.28.163.21:/home/shamil/thesis_sketch/ \
	--exclude .logs \
	--exclude .git \
	--exclude data \
	--exclude __pycache__ \
	--delete-after


connect: move_to_server
	ssh -t shamil@172.28.163.21 "cd thesis_sketch ; bash --login"




pre-train:
	poetry install



# start training & detach from console
train_bm25: pre-train
	rm -f logs/bm25/train.log
	screen -L -Logfile logs/bm25/train.log -dm bash -c "poetry run python src/train/train_bm25.py"
	sleep 3
	tail -f logs/bm25/train.log


train_bert: pre-train
	poetry run python src/pipelines/


lint:
	poetry run ruff --fix src/ --no-cache
	poetry run black src/
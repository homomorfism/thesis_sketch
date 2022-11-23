move_to_server:
	rsync -avz --exclude-from=.gitignore ./ shamil@10.241.1.217:/home/shamil/thesis_sketch/



lint:
	flake8 --max-line-length 100 --exit-zero src/
	isort src/


connect: move_to_server
	ssh -t shamil@10.241.1.217 "cd thesis_sketch ; bash --login"




# start trainining & detach from console
train_bm25:
	@rm -f logs/bm25.log
	screen -L -Logfile logs/bm25.log -dm bash -c "PYTHONPATH=. python -W ignore src/train/train_bm25.py"
	-@tail -f logs/bm25.log

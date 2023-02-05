"""Run training with only one command"""

import subprocess

import click


def run(saving_dir: str, machine: str, command: str):
    subprocess.run(['rsync', '-avz', "./", f"{machine}:{saving_dir}"])

    subprocess.run(['ssh', ])


@click.option("--saving-dir", type=str, default='/home/shamil/thesis_sketch/')
@click.option("--remote-machine", default='shamil@172.28.163.21')
@click.option("--command", type=str, required=True)
def main(saving_dir: str, remote_machine: str, command: str):
    run(saving_dir, remote_machine, command)

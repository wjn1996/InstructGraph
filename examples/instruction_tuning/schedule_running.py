import schedule
import time
import subprocess

def run_bash_script():
    # 执行你的 Bash 脚本的命令
    subprocess.run(["bash", "/home/jiw203/wjn/InstructGraph/examples/instruction_tuning/run_llama2_fsdp_flashattn.sh"])
    # 取消定时任务，确保它只执行一次
    job.cancel()

# 设置定时任务，在每天早上7点执行
job = schedule.every().day.at("07:00").do(run_bash_script)

num = 0
while not job.job_done:
    num += 1
    # 检查是否有定时任务需要执行
    schedule.run_pending()
    time.sleep(1)  # 等待一秒钟，避免占用太多系统资源
    if num % 1800 == 0:
        print("waiting for {} minute ...".format(int(num / 60)))

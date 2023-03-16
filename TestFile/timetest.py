import time

path = '../RollerAgent/saved_model/'

print(time.time())
t = time.time()   # 时间戳
print('当前时间戳：', t)
local_time = time.localtime(t)   # 当前时间
print('当前时间：',local_time)
asc_time = time.asctime(local_time)  # 格式化时间
print("格式化的时间：",asc_time)
format_time = time.strftime("%Y-%m-%d %H:%M:%S",local_time)   # 格式化日期
print("格式化日期：",format_time)

print(path + str(time.time())[:10] +'_actor.pkl')
kill -9 $(ps -ef|grep -E 'mypython'|grep -v grep|awk '{print $2}')

import smtplib, indexer, sys
from StringIO import StringIO
from datetime import datetime
#SERVER = "localhost"

try:
	FROM ='vs_report@vecshare.com'
	TO = ["fern.jared@gmail.com"] # must be a list
	output = StringIO()
	sys.stdout = output
	if(indexer.refresh()):
		SUBJECT = "Successful Index: New Embedding Uploaded"
		TEXT = "Indexed on: " + str(datetime.now()) + "\n"+output.getvalue()
	else:
		SUBJECT = "Successful Index: No Changes"
		TEXT = "Indexed on: " + str(datetime.now()) + "\n"+ output.getvalue()
except Exception, e:
	SUBJECT = "VecShare: Indexing Error"
	TEXT = "Error thrown at: " + str(datetime.now()) + "\n" + str(e)

message = """\
Subject: %s
%s
""" % (SUBJECT, TEXT)

server = smtplib.SMTP('localhost')
server.sendmail(FROM, TO, message)
server.quit()
sys.stdout.close()

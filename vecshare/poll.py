import smtplib
import indexer
from datetime import datetime
#SERVER = "localhost"

try:
	if(indexer.refresh()):
		FROM ='vs_report@vecshare.com'
		TO = ["fern.jared@gmail.com"] # must be a list
		SUBJECT = "Successful Index: New Embedding Uploaded"
		TEXT = "Indexed on: " + str(datetime.now())
		message = """ \
		From: %s
		To: %s
		Subject: %s
		%s
		""" % (FROM, ", ".join(TO), SUBJECT, TEXT)
		server = smtplib.SMTP('localhost')
		server.sendmail(FROM, TO, message)
		server.quit()
	else:
		FROM ='vs_report@vecshare.com'
		TO = ["fern.jared@gmail.com"] # must be a list
		SUBJECT = "Successful Index: No Changes"
		TEXT = "Indexed on: " + str(datetime.now())
		message = """ \
		From: %s
		To: %s
		Subject: %s
		%s
		""" % (FROM, ", ".join(TO), SUBJECT, TEXT)
		server = smtplib.SMTP('localhost')
		server.sendmail(FROM, TO, message)
		server.quit()
except:
	FROM = 'vs_report@vecshare.com'
	TO = ["fern.jared@gmail.com"] # must be a list

	SUBJECT = "VecShare: Indexing Error"

	TEXT = "Error thrown at: " + str(datetime.now())

	# Prepare actual message

	message = """\
	From: %s
	To: %s
	Subject: %s
	%s
	""" % (FROM, ", ".join(TO), SUBJECT, TEXT)

	# Send the mail

	server = smtplib.SMTP('localhost')
	server.sendmail(FROM, TO, message)
	server.quit()

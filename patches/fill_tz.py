'''Use timezonedb to get timezone information.  Save new queries to the metadata
   addenum'''
from __future__ import print_function
import csv
import json
import sqlite3
import sys
from shapely import wkb
import time
import traceback
import urllib
import urllib2

APIKEY="BWXDREGGGTRS"
con = sqlite3.connect(":memory:")
cur = con.cursor()
# Load the timezone addendum once into sqlite memory
TZ_FIELDS = ['abbreviation', 'countryCode', 'nextAbbreviation', 'timestamp',
             'dst', 'dstStart', 'countryName', 'gmtOffset', 'dstEnd', 'zoneName']
cur.execute("CREATE TABLE timezone (%s)"%(','.join(["site_id UNIQUE"] + TZ_FIELDS)))
con.commit()
try:
    with open("site_timezone.csv", "rb") as tz_file:
        site_tz = csv.reader(tz_file)
        fields = next(site_tz)
        #fields[0] = fields[0]+" UNIQUE"
        #print "CREATE TABLE timezone (%s)"%(",".join(fields))
        #cur.execute("CREATE TABLE timezone (%s)"%(",".join(fields)))
        for row in site_tz:
            cur.execute("INSERT OR IGNORE INTO timezone (%s) VALUES (%s)"%(", ".join(fields), ",".join(["?" for f in fields])), row)
            con.commit()
except:
    print( "No site timezone data")
    traceback.print_exc()


# Load the site metadata once
with open("three_tier_site_metadata.csv", "rb") as site_file:
    site_csv = csv.reader(site_file)
    fields = next(site_csv)
    cur.execute("CREATE TABLE sites (%s)"%(",".join(fields)))
    for row in site_csv:
        cur.execute("INSERT INTO sites (%s) VALUES (%s)"%(",".join(fields), ",".join(["?" for f in fields])), row)
    con.commit()

def check_addendum_for_tz(site_id):
    '''Check if addendum has the tz
    '''

def get_tz_for_site(site_id):
    '''Return the timezone information for site_id, soring desired fields to
       timezone database
    '''
    #print "Getting timezone for site_id %s"%site_id
    cur.execute("SELECT the_geom FROM sites WHERE site_id=?", [str(site_id)])
    tzdat = cur.fetchall()
    #print tzdat
    point = wkb.loads(tzdat[0][0], hex=True)
    #print "Site %s loc: (%s, %s)"%(site_id, point.x, point.y)
    url = "http://api.timezonedb.com/v2/get-time-zone?"
    data = {'key':APIKEY, 'format':'json', 'by':'position', 'lat':point.y, 'lng':point.x}
    url += urllib.urlencode(data)
    req = urllib2.Request(url, urllib.urlencode(data))
    start_time = time.time()
    try:
        res = urllib2.urlopen(req)
    except:
        sys.stderr.write('e')
        sys.stderr.flush()
        time.sleep(30)
        return {}
    req_time = time.time() - start_time
    res_data = res.read()
    if req_time > 60:
        sys.stderr.write('v')
        sys.stderr.flush()
    elif req_time > 10:
        sys.stderr.write('l')
        sys.stderr.flush()
        #print "\nLong request time %d seconds: message %s"%(req_time, res_data)
    try:
        result = json.loads(res_data)
        if result["status"] != "OK":
            raise Exception("Bad timezone request, result was %s"%(result["status"]))
        cur.execute("INSERT OR IGNORE INTO timezone (%s) VALUES (%s)"%(",".join(["site_id"]+TZ_FIELDS), ",".join(["'%s'"%site_id]+["'%s'"%result[f] for f in TZ_FIELDS])))
        con.commit()
    except:
        print( "Bad data")
        traceback.print_exc()
        raise Exception
    return result


def save_tz_to_file(filename):
    '''Save the timezone database to file
    '''
    with open(filename, 'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["site_id"]+TZ_FIELDS)
        cur.execute("SELECT %s FROM timezone ORDER BY CAST(site_id AS INTEGER)"%(",".join(['site_id']+TZ_FIELDS)))
        for row in cur.fetchall():
            writer.writerow(row)

def fill_tz(filename):
    '''Fill the timezone database for missing site_ids and write to filename'''
    cur.execute("""SELECT site_id FROM sites
                   WHERE site_id NOT IN (SELECT site_id FROM timezone)
                   ORDER BY CAST(site_id AS INTEGER) ASC""")
    missing = [m[0] for m in cur.fetchall()]
    #print "Missing %s sites"%len(missing)
    print( "Missing {} sites".format(len(missing)))
    sys.stdout.write('Starting queries')
    sys.stdout.flush()
    try:
        # Just do 5
        count = 1
        for s in missing:
            if count%100 == 0:
                sys.stdout.write('w')
                sys.stdout.flush()
                save_tz_to_file(filename)
            elif count%5 == 0:
                sys.stdout.write('.')
                sys.stdout.flush()
            try:
                get_tz_for_site(s)
            except:
                print( "Problem with get_tz_for_site")
                traceback.print_exc()
                break
            time.sleep(1.1) # API limit is 1 per second, playing it safe to avoid lockout
            count += 1
    except:
        print( "Problem with fill_tz")
        traceback.print_exc()
    finally:
        save_tz_to_file(filename)
    #print "\nPulled %s sites"%count
    print("\nPulled {} sites".format(count))

def main():
    #get_tz_for_site(sys.argv[1])
    fill_tz('site_timezone.csv')

if __name__ == "__main__":
    main()

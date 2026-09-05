from app.providers.treasury import parse_yield_curve_xml


SAMPLE_XML = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns:d="http://schemas.microsoft.com/ado/2007/08/dataservices"
      xmlns:m="http://schemas.microsoft.com/ado/2007/08/dataservices/metadata"
      xmlns="http://www.w3.org/2005/Atom">
<entry>
<content type="application/xml">
<m:properties>
<d:NEW_DATE m:type="Edm.DateTime">2026-09-03T00:00:00</d:NEW_DATE>
<d:BC_2YEAR m:type="Edm.Double">4.34</d:BC_2YEAR>
<d:BC_10YEAR m:type="Edm.Double">4.77</d:BC_10YEAR>
</m:properties>
</content>
</entry>
</feed>
"""


def test_parse_yield_curve_xml_extracts_2y_and_10y():
    frame = parse_yield_curve_xml(SAMPLE_XML)
    assert frame.iloc[0]["date"].year == 2026
    assert frame.iloc[0]["US_2Y"] == 4.34
    assert frame.iloc[0]["US_10Y"] == 4.77

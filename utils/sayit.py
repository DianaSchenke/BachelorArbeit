#source: https://github.com/hal3/sirewrite/blob/master/sayit.py

import re,sys

def sayWord(lang, s):
    if len(s) == 0: return 0
    elif lang == 'de': return 0.201443 + 0.247255*len(re.findall(u'[0-9]{2}',s)) + 0.220354*len(re.findall(u'[0-9]',s)) + 0.155791*len(re.findall(u'q',s)) + 0.147522*len(re.findall(u'x',s)) + 0.141111*len(re.findall(u'[0-9]{4}',s)) + 0.111575*len(re.findall(u'\u00ee',s)) + 0.109382*len(re.findall(u'[^a\u00e2\u00e1\u00e0\u00e4\u01ce\u0103\u00e3\u0101\u00e6\u01e3e\u00ea\u00e9\u00e8\u00eb\u011b\u0115\u0113i\u00ef\u00ed\u00ee\u01d0\u012d\u0129\u0131\u012bo\u00f6\u00f3\u00f4\u00f6\u01d2\u014f\u00f5u\u00fa\u00f9\u00fbu\u01d4\u016d\u0169\u016by\u00fdbcdfghjklmnpqrstvwxz\u00f1\u00e7\u0159]',s)) + 0.099467*len(re.findall(u'o',s)) + 0.094990*len(re.findall(u'a',s)) + 0.089802*len(re.findall(u'z',s)) + 0.088050*len(re.findall(u'\u00f3',s)) + 0.080506*len(re.findall(u'\u00e1',s)) + 0.079824*len(re.findall(u'\u00e2',s)) + 0.078788*len(re.findall(u'y',s)) + 0.073479*len(re.findall(u'\u00e0',s)) + 0.068338*len(re.findall(u'\u00e4',s)) + 0.066598*len(re.findall(u's',s)) + 0.063880*len(re.findall(u'\u00ef',s)) + 0.062730*len(re.findall(u'i',s)) + 0.061144*len(re.findall(u'k',s)) + 0.057448*len(re.findall(u'\u00f4',s)) + 0.055320*len(re.findall(u'\u00e9',s)) + 0.055251*len(re.findall(u'\u00e8',s)) + 0.053560*len(re.findall(u'v',s)) + 0.052198*len(re.findall(u'\u00ed',s)) + 0.050962*len(re.findall(u'\u00ea',s)) + 0.049994*len(re.findall(u'b',s)) + 0.049982*len(re.findall(u'f',s)) + 0.047746*len(re.findall(u'e',s)) + 0.047723*len(re.findall(u'^[a\u00e2\u00e1\u00e0\u00e4\u01ce\u0103\u00e3\u0101\u00e6\u01e3e\u00ea\u00e9\u00e8\u00eb\u011b\u0115\u0113i\u00ef\u00ed\u00ee\u01d0\u012d\u0129\u0131\u012bo\u00f6\u00f3\u00f4\u00f6\u01d2\u014f\u00f5u\u00fa\u00f9\u00fbu\u01d4\u016d\u0169\u016by\u00fd]',s)) + 0.047120*len(re.findall(u'j',s)) + 0.046027*len(re.findall(u'p',s)) + 0.045183*len(re.findall(u't',s)) + 0.040199*len(re.findall(u'r',s)) + 0.039382*len(re.findall(u'd',s)) + 0.037563*len(re.findall(u'\u00f1',s)) + 0.037290*len(re.findall(u'\u00f6',s)) + 0.037290*len(re.findall(u'\u00f6',s)) + 0.036302*len(re.findall(u'^[bcdfghjklmnpqrstvwxz\u00f1\u00e7\u0159]{3}',s)) + 0.035693*len(re.findall(u'g',s)) + 0.034323*len(re.findall(u'\u00fb',s)) + 0.032789*len(re.findall(u'u',s)) + 0.032789*len(re.findall(u'u',s)) + 0.032587*len(re.findall(u'[a\u00e2\u00e1\u00e0\u00e4\u01ce\u0103\u00e3\u0101\u00e6\u01e3e\u00ea\u00e9\u00e8\u00eb\u011b\u0115\u0113i\u00ef\u00ed\u00ee\u01d0\u012d\u0129\u0131\u012bo\u00f6\u00f3\u00f4\u00f6\u01d2\u014f\u00f5u\u00fa\u00f9\u00fbu\u01d4\u016d\u0169\u016by\u00fd][bcdfghjklmnpqrstvwxz\u00f1\u00e7\u0159]',s)) + 0.030861*len(re.findall(u'n',s)) + 0.029051*len(re.findall(u'm',s)) + 0.027110*len(re.findall(u'l',s)) + 0.027075*len(re.findall(u'w',s)) + 0.025398*len(re.findall(u'h',s)) + 0.023578*len(re.findall(u'\u00e7',s)) + 0.013630*len(re.findall(u'[a\u00e2\u00e1\u00e0\u00e4\u01ce\u0103\u00e3\u0101\u00e6\u01e3e\u00ea\u00e9\u00e8\u00eb\u011b\u0115\u0113i\u00ef\u00ed\u00ee\u01d0\u012d\u0129\u0131\u012bo\u00f6\u00f3\u00f4\u00f6\u01d2\u014f\u00f5u\u00fa\u00f9\u00fbu\u01d4\u016d\u0169\u016by\u00fd]\$',s)) + 0.012129*len(re.findall(u'c',s)) + 0.002293*len(re.findall(u'\u00eb',s))
    elif lang == 'en-US': return 0.288128 + 0.954096*len(re.findall(u'[0-9]{4}',s)) + 0.356994*len(re.findall(u'[0-9]',s)) + 0.289716*len(re.findall(u'\u0169',s)) + 0.260762*len(re.findall(u'^[bcdfghjklmnpqrstvwxz\u00f1\u00e7\u0159]{3}',s)) + 0.202507*len(re.findall(u'\u00f1',s)) + 0.189280*len(re.findall(u'q',s)) + 0.172014*len(re.findall(u'x',s)) + 0.138827*len(re.findall(u'n',s)) + 0.132514*len(re.findall(u'm',s)) + 0.128558*len(re.findall(u'\u012b',s)) + 0.117963*len(re.findall(u'd',s)) + 0.115260*len(re.findall(u'\u00ed',s)) + 0.111563*len(re.findall(u'b',s)) + 0.109092*len(re.findall(u'v',s)) + 0.107073*len(re.findall(u'\u00e7',s)) + 0.105800*len(re.findall(u'p',s)) + 0.104254*len(re.findall(u'l',s)) + 0.100262*len(re.findall(u'r',s)) + 0.099480*len(re.findall(u'j',s)) + 0.097076*len(re.findall(u't',s)) + 0.096195*len(re.findall(u'c',s)) + 0.095999*len(re.findall(u'\u00fa',s)) + 0.090362*len(re.findall(u'[0-9]{2}',s)) + 0.082615*len(re.findall(u'\u016b',s)) + 0.080488*len(re.findall(u's',s)) + 0.075311*len(re.findall(u'g',s)) + 0.075048*len(re.findall(u'z',s)) + 0.067591*len(re.findall(u'k',s)) + 0.065711*len(re.findall(u'f',s)) + 0.065148*len(re.findall(u'[a\u00e2\u00e1\u00e0\u00e4\u01ce\u0103\u00e3\u0101\u00e6\u01e3e\u00ea\u00e9\u00e8\u00eb\u011b\u0115\u0113i\u00ef\u00ed\u00ee\u01d0\u012d\u0129\u0131\u012bo\u00f6\u00f3\u00f4\u00f6\u01d2\u014f\u00f5u\u00fa\u00f9\u00fbu\u01d4\u016d\u0169\u016by\u00fd][bcdfghjklmnpqrstvwxz\u00f1\u00e7\u0159]',s)) + 0.063661*len(re.findall(u'a',s)) + 0.060130*len(re.findall(u'y',s)) + 0.059559*len(re.findall(u'\u0159',s)) + 0.056398*len(re.findall(u'i',s)) + 0.054900*len(re.findall(u'o',s)) + 0.050908*len(re.findall(u'\u00e1',s)) + 0.048779*len(re.findall(u'w',s)) + 0.046645*len(re.findall(u'\u00fb',s)) + 0.034323*len(re.findall(u'\u00e2',s)) + 0.033232*len(re.findall(u'\u00eb',s)) + 0.031705*len(re.findall(u'\u00f4',s)) + 0.029573*len(re.findall(u'\u0101',s)) + 0.028180*len(re.findall(u'u',s)) + 0.028180*len(re.findall(u'u',s)) + 0.027809*len(re.findall(u'\u00e4',s)) + 0.026002*len(re.findall(u'\u00f3',s)) + 0.024175*len(re.findall(u'[a\u00e2\u00e1\u00e0\u00e4\u01ce\u0103\u00e3\u0101\u00e6\u01e3e\u00ea\u00e9\u00e8\u00eb\u011b\u0115\u0113i\u00ef\u00ed\u00ee\u01d0\u012d\u0129\u0131\u012bo\u00f6\u00f3\u00f4\u00f6\u01d2\u014f\u00f5u\u00fa\u00f9\u00fbu\u01d4\u016d\u0169\u016by\u00fd]\$',s)) + 0.021481*len(re.findall(u'e',s)) + 0.019389*len(re.findall(u'\u0131',s)) + 0.018713*len(re.findall(u'[^a\u00e2\u00e1\u00e0\u00e4\u01ce\u0103\u00e3\u0101\u00e6\u01e3e\u00ea\u00e9\u00e8\u00eb\u011b\u0115\u0113i\u00ef\u00ed\u00ee\u01d0\u012d\u0129\u0131\u012bo\u00f6\u00f3\u00f4\u00f6\u01d2\u014f\u00f5u\u00fa\u00f9\u00fbu\u01d4\u016d\u0169\u016by\u00fdbcdfghjklmnpqrstvwxz\u00f1\u00e7\u0159]',s)) + 0.012112*len(re.findall(u'\u00e0',s)) + 0.010956*len(re.findall(u'h',s)) + 0.007262*len(re.findall(u'\u00ee',s)) + 0.004981*len(re.findall(u'\u00e8',s))
    elif lang == 'euro': return 0.187036 + 0.317005*len(re.findall(u'\u00e3',s)) + 0.315815*len(re.findall(u'\u0169',s)) + 0.292691*len(re.findall(u'\u0103',s)) + 0.253129*len(re.findall(u'\u011b',s)) + 0.227771*len(re.findall(u'[^a\u00e2\u00e1\u00e0\u00e4\u01ce\u0103\u00e3\u0101\u00e6\u01e3e\u00ea\u00e9\u00e8\u00eb\u011b\u0115\u0113i\u00ef\u00ed\u00ee\u01d0\u012d\u0129\u0131\u012bo\u00f6\u00f3\u00f4\u00f6\u01d2\u014f\u00f5u\u00fa\u00f9\u00fbu\u01d4\u016d\u0169\u016by\u00fdbcdfghjklmnpqrstvwxz\u00f1\u00e7\u0159]',s)) + 0.207797*len(re.findall(u'\u0159',s)) + 0.189736*len(re.findall(u'[0-9]{2}',s)) + 0.184000*len(re.findall(u'\u00e6',s)) + 0.182175*len(re.findall(u'\u012b',s)) + 0.167399*len(re.findall(u'\u00f5',s)) + 0.139567*len(re.findall(u'\u0101',s)) + 0.111823*len(re.findall(u'x',s)) + 0.102212*len(re.findall(u'\u016b',s)) + 0.099673*len(re.findall(u'k',s)) + 0.099467*len(re.findall(u'\u00fd',s)) + 0.097008*len(re.findall(u'z',s)) + 0.090673*len(re.findall(u'y',s)) + 0.077894*len(re.findall(u'\u0131',s)) + 0.075144*len(re.findall(u'p',s)) + 0.068023*len(re.findall(u'j',s)) + 0.064888*len(re.findall(u'f',s)) + 0.064057*len(re.findall(u't',s)) + 0.062861*len(re.findall(u'c',s)) + 0.061194*len(re.findall(u's',s)) + 0.059466*len(re.findall(u'o',s)) + 0.059261*len(re.findall(u'd',s)) + 0.058316*len(re.findall(u'w',s)) + 0.057997*len(re.findall(u'g',s)) + 0.052533*len(re.findall(u'a',s)) + 0.051308*len(re.findall(u'r',s)) + 0.050576*len(re.findall(u'i',s)) + 0.048901*len(re.findall(u'b',s)) + 0.047846*len(re.findall(u'v',s)) + 0.047137*len(re.findall(u'q',s)) + 0.045976*len(re.findall(u'[0-9]{4}',s)) + 0.043086*len(re.findall(u'n',s)) + 0.042979*len(re.findall(u'm',s)) + 0.040604*len(re.findall(u'\u00e4',s)) + 0.039670*len(re.findall(u'h',s)) + 0.039339*len(re.findall(u'l',s)) + 0.038610*len(re.findall(u'[a\u00e2\u00e1\u00e0\u00e4\u01ce\u0103\u00e3\u0101\u00e6\u01e3e\u00ea\u00e9\u00e8\u00eb\u011b\u0115\u0113i\u00ef\u00ed\u00ee\u01d0\u012d\u0129\u0131\u012bo\u00f6\u00f3\u00f4\u00f6\u01d2\u014f\u00f5u\u00fa\u00f9\u00fbu\u01d4\u016d\u0169\u016by\u00fd][bcdfghjklmnpqrstvwxz\u00f1\u00e7\u0159]',s)) + 0.035417*len(re.findall(u'e',s)) + 0.032111*len(re.findall(u'\u00e1',s)) + 0.030724*len(re.findall(u'^[bcdfghjklmnpqrstvwxz\u00f1\u00e7\u0159]{3}',s)) + 0.023949*len(re.findall(u'[a\u00e2\u00e1\u00e0\u00e4\u01ce\u0103\u00e3\u0101\u00e6\u01e3e\u00ea\u00e9\u00e8\u00eb\u011b\u0115\u0113i\u00ef\u00ed\u00ee\u01d0\u012d\u0129\u0131\u012bo\u00f6\u00f3\u00f4\u00f6\u01d2\u014f\u00f5u\u00fa\u00f9\u00fbu\u01d4\u016d\u0169\u016by\u00fd]\$',s)) + 0.021964*len(re.findall(u'\u00f6',s)) + 0.021964*len(re.findall(u'\u00f6',s)) + 0.014352*len(re.findall(u'u',s)) + 0.014352*len(re.findall(u'u',s)) + 0.000250*len(re.findall(u'[0-9]',s))
    elif lang == 'fr': return 0.094123 + 0.524550*len(re.findall(u'[0-9]{4}',s)) + 0.146372*len(re.findall(u'[0-9]{2}',s)) + 0.115989*len(re.findall(u'\u00ef',s)) + 0.075470*len(re.findall(u'j',s)) + 0.071641*len(re.findall(u'w',s)) + 0.067887*len(re.findall(u'[a\u00e2\u00e1\u00e0\u00e4\u01ce\u0103\u00e3\u0101\u00e6\u01e3e\u00ea\u00e9\u00e8\u00eb\u011b\u0115\u0113i\u00ef\u00ed\u00ee\u01d0\u012d\u0129\u0131\u012bo\u00f6\u00f3\u00f4\u00f6\u01d2\u014f\u00f5u\u00fa\u00f9\u00fbu\u01d4\u016d\u0169\u016by\u00fd]\$',s)) + 0.066211*len(re.findall(u'[a\u00e2\u00e1\u00e0\u00e4\u01ce\u0103\u00e3\u0101\u00e6\u01e3e\u00ea\u00e9\u00e8\u00eb\u011b\u0115\u0113i\u00ef\u00ed\u00ee\u01d0\u012d\u0129\u0131\u012bo\u00f6\u00f3\u00f4\u00f6\u01d2\u014f\u00f5u\u00fa\u00f9\u00fbu\u01d4\u016d\u0169\u016by\u00fd][bcdfghjklmnpqrstvwxz\u00f1\u00e7\u0159]',s)) + 0.064186*len(re.findall(u'g',s)) + 0.062926*len(re.findall(u'k',s)) + 0.060646*len(re.findall(u'y',s)) + 0.059996*len(re.findall(u'x',s)) + 0.059590*len(re.findall(u'f',s)) + 0.057422*len(re.findall(u'c',s)) + 0.055183*len(re.findall(u'\u00e7',s)) + 0.054494*len(re.findall(u'r',s)) + 0.053814*len(re.findall(u'z',s)) + 0.051429*len(re.findall(u't',s)) + 0.049218*len(re.findall(u's',s)) + 0.044615*len(re.findall(u'\u00eb',s)) + 0.044090*len(re.findall(u'i',s)) + 0.043803*len(re.findall(u'd',s)) + 0.042808*len(re.findall(u'\u00f6',s)) + 0.042808*len(re.findall(u'\u00f6',s)) + 0.042791*len(re.findall(u'b',s)) + 0.042591*len(re.findall(u'q',s)) + 0.039526*len(re.findall(u'p',s)) + 0.038798*len(re.findall(u'v',s)) + 0.035982*len(re.findall(u'\u00e9',s)) + 0.034998*len(re.findall(u'o',s)) + 0.032743*len(re.findall(u'[0-9]',s)) + 0.031503*len(re.findall(u'a',s)) + 0.028726*len(re.findall(u'[^a\u00e2\u00e1\u00e0\u00e4\u01ce\u0103\u00e3\u0101\u00e6\u01e3e\u00ea\u00e9\u00e8\u00eb\u011b\u0115\u0113i\u00ef\u00ed\u00ee\u01d0\u012d\u0129\u0131\u012bo\u00f6\u00f3\u00f4\u00f6\u01d2\u014f\u00f5u\u00fa\u00f9\u00fbu\u01d4\u016d\u0169\u016by\u00fdbcdfghjklmnpqrstvwxz\u00f1\u00e7\u0159]',s)) + 0.027647*len(re.findall(u'h',s)) + 0.027537*len(re.findall(u'm',s)) + 0.027147*len(re.findall(u'l',s)) + 0.025762*len(re.findall(u'\u00f4',s)) + 0.020906*len(re.findall(u'^[bcdfghjklmnpqrstvwxz\u00f1\u00e7\u0159]{3}',s)) + 0.020085*len(re.findall(u'\u00e8',s)) + 0.016057*len(re.findall(u'n',s)) + 0.014943*len(re.findall(u'\u00e2',s)) + 0.011852*len(re.findall(u'u',s)) + 0.011852*len(re.findall(u'u',s)) + 0.008881*len(re.findall(u'\u00e0',s)) + 0.006535*len(re.findall(u'e',s)) + 0.003959*len(re.findall(u'\u00ea',s))
    elif lang == 'it': return 0.216031 + 0.280613*len(re.findall(u'[0-9]{2}',s)) + 0.219277*len(re.findall(u'x',s)) + 0.149924*len(re.findall(u'[0-9]',s)) + 0.145241*len(re.findall(u'[^a\u00e2\u00e1\u00e0\u00e4\u01ce\u0103\u00e3\u0101\u00e6\u01e3e\u00ea\u00e9\u00e8\u00eb\u011b\u0115\u0113i\u00ef\u00ed\u00ee\u01d0\u012d\u0129\u0131\u012bo\u00f6\u00f3\u00f4\u00f6\u01d2\u014f\u00f5u\u00fa\u00f9\u00fbu\u01d4\u016d\u0169\u016by\u00fdbcdfghjklmnpqrstvwxz\u00f1\u00e7\u0159]',s)) + 0.115833*len(re.findall(u'z',s)) + 0.104811*len(re.findall(u'c',s)) + 0.104514*len(re.findall(u'p',s)) + 0.101104*len(re.findall(u't',s)) + 0.096837*len(re.findall(u'k',s)) + 0.094215*len(re.findall(u'q',s)) + 0.092929*len(re.findall(u'f',s)) + 0.091644*len(re.findall(u's',s)) + 0.075643*len(re.findall(u'n',s)) + 0.073194*len(re.findall(u'g',s)) + 0.071579*len(re.findall(u'y',s)) + 0.067655*len(re.findall(u'j',s)) + 0.066069*len(re.findall(u'b',s)) + 0.064722*len(re.findall(u'e',s)) + 0.059960*len(re.findall(u'v',s)) + 0.058603*len(re.findall(u'i',s)) + 0.057001*len(re.findall(u'm',s)) + 0.054888*len(re.findall(u'w',s)) + 0.053622*len(re.findall(u'o',s)) + 0.052965*len(re.findall(u'd',s)) + 0.052437*len(re.findall(u'a',s)) + 0.051798*len(re.findall(u'l',s)) + 0.051058*len(re.findall(u'\u00e9',s)) + 0.050466*len(re.findall(u'\u00e1',s)) + 0.049816*len(re.findall(u'r',s)) + 0.049440*len(re.findall(u'\u00f9',s)) + 0.048566*len(re.findall(u'\u00fa',s)) + 0.044918*len(re.findall(u'\u00e2',s)) + 0.034806*len(re.findall(u'\u00f1',s)) + 0.031362*len(re.findall(u'u',s)) + 0.031362*len(re.findall(u'u',s)) + 0.028077*len(re.findall(u'\u00f3',s)) + 0.023515*len(re.findall(u'\u00ea',s)) + 0.013186*len(re.findall(u'\u00f4',s))
    elif lang == 'ja': return 0.192934 + 0.401897*len(re.findall(u'\uff57',s)) + 0.391781*len(re.findall(u'\uff37',s)) + 0.345174*len(re.findall(u'\uff16',s)) + 0.337752*len(re.findall(u'\uff17',s)) + 0.335693*len(re.findall(u'\uff18',s)) + 0.305787*len(re.findall(u'\uff19',s)) + 0.284248*len(re.findall(u'\uff13',s)) + 0.260788*len(re.findall(u'\uff14',s)) + 0.250454*len(re.findall(u'\uff32',s)) + 0.249550*len(re.findall(u'\uff52',s)) + 0.239987*len(re.findall(u'\uff48',s)) + 0.232954*len(re.findall(u'\uff15',s)) + 0.224712*len(re.findall(u'\uff28',s)) + 0.201113*len(re.findall(u'\uff4d',s)) + 0.198311*len(re.findall(u'\uff56',s)) + 0.197385*len(re.findall(u'\uff12',s)) + 0.195817*len(re.findall(u'\uff23',s)) + 0.194144*len(re.findall(u'\uff2c',s)) + 0.191657*len(re.findall(u'\uff47',s)) + 0.187868*len(re.findall(u'\uff4e',s)) + 0.183400*len(re.findall(u'\uff27',s)) + 0.181215*len(re.findall(u'\uff50',s)) + 0.177365*len(re.findall(u'\uff2d',s)) + 0.175796*len(re.findall(u'\uff34',s)) + 0.175779*len(re.findall(u'\uff11',s)) + 0.174960*len(re.findall(u'\uff42',s)) + 0.172881*len(re.findall(u'\uff49',s)) + 0.171703*len(re.findall(u'\uff43',s)) + 0.171623*len(re.findall(u'\uff4c',s)) + 0.168697*len(re.findall(u'\uff53',s)) + 0.166248*len(re.findall(u'\uff54',s)) + 0.162419*len(re.findall(u'\uff25',s)) + 0.161625*len(re.findall(u'\uff2a',s)) + 0.159372*len(re.findall(u'\uff22',s)) + 0.158361*len(re.findall(u'\uff55',s)) + 0.158059*len(re.findall(u'\uff24',s)) + 0.156341*len(re.findall(u'\uff41',s)) + 0.153941*len(re.findall(u'\uff45',s)) + 0.153059*len(re.findall(u'\uff39',s)) + 0.149345*len(re.findall(u'\uff33',s)) + 0.147929*len(re.findall(u'\uff2e',s)) + 0.146199*len(re.findall(u'\uff46',s)) + 0.144357*len(re.findall(u'\uff35',s)) + 0.144285*len(re.findall(u'\uff44',s)) + 0.134242*len(re.findall(u'\uff21',s)) + 0.132635*len(re.findall(u'\uff26',s)) + 0.129675*len(re.findall(u'\uff29',s)) + 0.122793*len(re.findall(u'\uff4b',s)) + 0.122485*len(re.findall(u'\uff2b',s)) + 0.122419*len(re.findall(u'\uff30',s)) + 0.122348*len(re.findall(u'\uff4f',s)) + 0.117883*len(re.findall(u'\uff36',s)) + 0.116234*len(re.findall(u'\uff59',s)) + 0.110487*len(re.findall(u'^[\u4E00-\u9FFF]',s)) + 0.104395*len(re.findall(u'\u304f',s)) + 0.100524*len(re.findall(u'\u30b6',s)) + 0.099742*len(re.findall(u'\u30ba',s)) + 0.095281*len(re.findall(u'\u5cf6',s)) + 0.094178*len(re.findall(u'\u30ab',s)) + 0.094091*len(re.findall(u'\u308f',s)) + 0.088825*len(re.findall(u'\u30bc',s)) + 0.086408*len(re.findall(u'\u30b5',s)) + 0.086084*len(re.findall(u'[\u4E00-\u9FFF]\$',s)) + 0.083627*len(re.findall(u'\u30dd',s)) + 0.082973*len(re.findall(u'\uff2f',s)) + 0.082004*len(re.findall(u'\u304c',s)) + 0.081355*len(re.findall(u'\u30bb',s)) + 0.080537*len(re.findall(u'\u56fd',s)) + 0.080233*len(re.findall(u'\u90ce',s)) + 0.076580*len(re.findall(u'\u30b9',s)) + 0.071712*len(re.findall(u'\u30db',s)) + 0.071397*len(re.findall(u'\u30cb',s)) + 0.070678*len(re.findall(u'\u30b3',s)) + 0.070540*len(re.findall(u'\u30cf',s)) + 0.069523*len(re.findall(u'\u305f',s)) + 0.066932*len(re.findall(u'\u30d1',s)) + 0.066558*len(re.findall(u'\u7acb',s)) + 0.066497*len(re.findall(u'\u30c8',s)) + 0.065144*len(re.findall(u'\u30bf',s)) + 0.065009*len(re.findall(u'\u308d',s)) + 0.064780*len(re.findall(u'\u3068',s)) + 0.064771*len(re.findall(u'\u3064',s)) + 0.063986*len(re.findall(u'\u30af',s)) + 0.062007*len(re.findall(u'\u5357',s)) + 0.061800*len(re.findall(u'\u30bd',s)) + 0.060136*len(re.findall(u'\u30e4',s)) + 0.059696*len(re.findall(u'\u307e',s)) + 0.059586*len(re.findall(u'\u30d9',s)) + 0.059112*len(re.findall(u'\u30ef',s)) + 0.058790*len(re.findall(u'\u3084',s)) + 0.058416*len(re.findall(u'\u30b0',s)) + 0.058322*len(re.findall(u'\u5ddd',s)) + 0.058256*len(re.findall(u'\u30ad',s)) + 0.058247*len(re.findall(u'\u30ea',s)) + 0.057527*len(re.findall(u'\u30e1',s)) + 0.057262*len(re.findall(u'\u30c9',s)) + 0.057210*len(re.findall(u'\u5c71',s)) + 0.056999*len(re.findall(u'\u30cc',s)) + 0.056670*len(re.findall(u'\u30e0',s)) + 0.056596*len(re.findall(u'\u30d3',s)) + 0.056482*len(re.findall(u'\u672c',s)) + 0.055207*len(re.findall(u'\u30d4',s)) + 0.055031*len(re.findall(u'\u30c0',s)) + 0.054831*len(re.findall(u'\u4e2d',s)) + 0.054312*len(re.findall(u'\u305b',s)) + 0.053996*len(re.findall(u'\u3053',s)) + 0.053672*len(re.findall(u'\u30b7',s)) + 0.053307*len(re.findall(u'\u30cd',s)) + 0.053183*len(re.findall(u'\u3061',s)) + 0.053151*len(re.findall(u'\u30ed',s)) + 0.053137*len(re.findall(u'\u30fc',s)) + 0.052671*len(re.findall(u'\u30b8',s)) + 0.052482*len(re.findall(u'\u30ce',s)) + 0.052469*len(re.findall(u'\u30df',s)) + 0.052218*len(re.findall(u'\u3059',s)) + 0.051149*len(re.findall(u'\u30d0',s)) + 0.051120*len(re.findall(u'\u30c1',s)) + 0.050576*len(re.findall(u'\u3055',s)) + 0.050530*len(re.findall(u'\u9ad8',s)) + 0.050406*len(re.findall(u'\u3051',s)) + 0.050285*len(re.findall(u'\u30d5',s)) + 0.049924*len(re.findall(u'\u30b2',s)) + 0.049910*len(re.findall(u'\u3057',s)) + 0.049552*len(re.findall(u'\u30d2',s)) + 0.048954*len(re.findall(u'\u304d',s)) + 0.048841*len(re.findall(u'\u30e2',s)) + 0.047662*len(re.findall(u'\u306a',s)) + 0.046810*len(re.findall(u'\u30d7',s)) + 0.046802*len(re.findall(u'\u306e',s)) + 0.046168*len(re.findall(u'\u3046',s)) + 0.045981*len(re.findall(u'\u30f3',s)) + 0.045687*len(re.findall(u'\u3089',s)) + 0.045390*len(re.findall(u'\u30ec',s)) + 0.045015*len(re.findall(u'\u30ac',s)) + 0.044927*len(re.findall(u'\u30da',s)) + 0.044909*len(re.findall(u'\u884c',s)) + 0.043977*len(re.findall(u'\u30de',s)) + 0.043935*len(re.findall(u'\u30f4',s)) + 0.043439*len(re.findall(u'\u30c4',s)) + 0.042776*len(re.findall(u'\u307f',s)) + 0.042713*len(re.findall(u'\u30ca',s)) + 0.041510*len(re.findall(u'\u30c3',s)) + 0.039911*len(re.findall(u'\u30e9',s)) + 0.039382*len(re.findall(u'\u30ae',s)) + 0.036680*len(re.findall(u'\u5317',s)) + 0.034881*len(re.findall(u'\u30b1',s)) + 0.034862*len(re.findall(u'\u30eb',s)) + 0.033453*len(re.findall(u'\u539f',s)) + 0.033265*len(re.findall(u'\u65b0',s)) + 0.031582*len(re.findall(u'^[\u3040-\u30FA\uFF66-\uFF9D]{3}',s)) + 0.031414*len(re.findall(u'\u795e',s)) + 0.030574*len(re.findall(u'\u30d6',s)) + 0.030494*len(re.findall(u'\u30dc',s)) + 0.030311*len(re.findall(u'\u3063',s)) + 0.030213*len(re.findall(u'\u30c6',s)) + 0.029947*len(re.findall(u'.',s)) + 0.029441*len(re.findall(u'[\u4E00-\u9FFF]',s)) + 0.027448*len(re.findall(u'\u3093',s)) + 0.026703*len(re.findall(u'\u3052',s)) + 0.026097*len(re.findall(u'\u308a',s)) + 0.025692*len(re.findall(u'\u3081',s)) + 0.024119*len(re.findall(u'\u3044',s)) + 0.023380*len(re.findall(u'\u30a2',s)) + 0.022242*len(re.findall(u'\u3058',s)) + 0.021137*len(re.findall(u'\u4e0a',s)) + 0.019239*len(re.findall(u'\u30a6',s)) + 0.018306*len(re.findall(u'\u30c7',s)) + 0.017776*len(re.findall(u'\u6b63',s)) + 0.017477*len(re.findall(u'\u9577',s)) + 0.013000*len(re.findall(u'\u5e73',s)) + 0.012673*len(re.findall(u'\u30b4',s)) + 0.009949*len(re.findall(u'\u6771',s)) + 0.009247*len(re.findall(u'\u897f',s)) + 0.006630*len(re.findall(u'\u30a8',s)) + 0.006460*len(re.findall(u'[^\u4E00-\u9FFF\u3040-\u30FA\uFF66-\uFF9D]',s))
    else: raise Exception("unknown language: " + lang)

def sayit(lang, words):
    time = -0.116751 + 0.099640 * len(words)
    for w in words:
        time += 0.691677 * sayWord(lang, w)
    if time < 0: time = 0
    return time

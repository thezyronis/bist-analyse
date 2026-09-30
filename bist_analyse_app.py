"""
BIST Technische Analyse
=======================

Streamlit-Anwendung zur technischen Analyse von Aktien und Indizes der Borsa İstanbul (BIST).

Funktionsumfang
---------------
* Kursdaten laden: Yahoo Finance (yfinance), optional borsapy/TradingView, eigene CSV-Datei, Demo-Daten
* Datenprüfung: fehlende Werte, Lücken, unplausible Kurssprünge, unvollständige letzte Kerze
* Indikatoren: SMA 20/50/200, EMA 12/26, Bollinger-Bänder, RSI 14, MACD (12/26/9), Stochastik,
  ATR 14, ADX 14, OBV, Volumen-Durchschnitt und relative Volumenstärke
* Trendanalyse, Punktesystem und Kauf-/Verkaufssignale ohne Look-Ahead-Bias
* Risikoanalyse: Volatilität, ATR-Stop-Loss, Kursziel per Chance-Risiko-Verhältnis, Positionsgröße,
  Verlustserien, Drawdown, Sharpe Ratio
* Backtest der Signalstrategie inkl. Transaktionskosten und Slippage, Vergleich mit Buy & Hold
* Interaktive Plotly-Charts, Textbericht, CSV-/HTML-Export und optionale SQLite-Speicherung

Start
-----
    streamlit run bist_analyse_app.py

Wichtiger Hinweis
-----------------
Diese Anwendung dient ausschließlich zu Informations- und Bildungszwecken. Die Signale stellen
keine Finanz-, Anlage- oder Steuerberatung dar.
"""

from __future__ import annotations

import difflib
import io
import math
import re
import sqlite3
import sys
import unicodedata
import warnings
import zlib
from collections.abc import Sequence
from contextlib import closing, suppress
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:  # Die Rechenfunktionen funktionieren auch ohne Streamlit (z. B. in Tests).
    import streamlit as st
except ImportError:  # pragma: no cover
    st = None  # type: ignore[assignment]


# =============================================================================
# 0) Konstanten und Grundeinstellungen
# =============================================================================

APP_NAME = "BIST Technische Analyse"
APP_VERSION = "1.0.0"

HAFTUNGSAUSSCHLUSS = (
    "Diese Anwendung dient ausschließlich zu Informations- und Bildungszwecken. "
    "Die Signale stellen keine Finanz-, Anlage- oder Steuerberatung dar."
)
DATENHINWEIS = (
    "Die Kursdaten stammen aus inoffiziellen Schnittstellen (z. B. Yahoo Finance) und können verzögert "
    "(häufig 15 Minuten oder mehr), unvollständig oder fehlerhaft sein. Bereinigungen um Splits, "
    "Bonusaktien (bedelsiz sermaye artırımı) und Dividenden werden je nach Anbieter unterschiedlich "
    "umgesetzt. Kurse vor jeder Entscheidung mit einer offiziellen Quelle (Borsa İstanbul, Broker) abgleichen."
)
BACKTEST_HINWEIS = (
    "Vergangene Ergebnisse sind keine Garantie für zukünftige Renditen. Der Backtest ist eine "
    "vereinfachte Simulation (keine Steuern, keine Teilausführungen, keine Liquiditätsgrenzen)."
)

ZEITZONE_BIST = "Europe/Istanbul"
BIST_SITZUNGSENDE_MIN = 18 * 60 + 10   # Ende der Schlussauktion (18:10 Uhr Istanbul-Zeit)
DATENPUFFER_MIN = 20                    # Puffer für die Verzögerung der Datenquelle
BIST_PREISGRENZE = 0.10                 # Tägliche Preisgrenze (Tavan/Taban) der meisten BIST-Aktien
KURSSPRUNG_WARNSCHWELLE = 0.15          # Tagesbewegungen darüber gelten als verdächtig
STANDARD_SUFFIX = ".IS"                 # Yahoo-Finance-Suffix für Borsa İstanbul
HANDELSTAGE_PRO_JAHR = 252

QUELLE_YAHOO = "Yahoo Finance (yfinance)"
QUELLE_BORSAPY = "borsapy / TradingView (optional)"
QUELLE_CSV = "Eigene CSV-Datei"
QUELLE_DEMO = "Demo-Daten (offline)"
DATENQUELLEN = [QUELLE_YAHOO, QUELLE_BORSAPY, QUELLE_CSV, QUELLE_DEMO]
LIVE_QUELLEN = {QUELLE_YAHOO, QUELLE_BORSAPY}

# Anzeigezeitraum in Kalendertagen
ZEITRAEUME: dict[str, int] = {
    "1 Monat": 31,
    "3 Monate": 92,
    "6 Monate": 183,
    "1 Jahr": 366,
    "3 Jahre": 1096,
    "5 Jahre": 1827,
}

# Währungsansicht: Wechselkurs-Symbol bei Yahoo Finance (TRY je Einheit der Zielwährung)
WAEHRUNGEN: dict[str, str | None] = {"TRY": None, "USD": "USDTRY=X", "EUR": "EURTRY=X"}


@dataclass(frozen=True)
class IntervallInfo:
    """Beschreibt ein Kerzenintervall und die Grenzen der Datenquelle."""

    code: str                  # Intervall-Code in yfinance-Notation
    intraday: bool             # True für Minuten-/Stundenkerzen
    minuten: int | None        # Kerzenlänge in Minuten (nur intraday)
    max_tage: int | None       # maximal abrufbare Historie in Kalendertagen (None = unbegrenzt)
    vorlauf_tage: int          # zusätzliche Kalendertage als Vorlauf für langsame Indikatoren (SMA 200)
    perioden_pro_jahr: float   # Näherung für die Annualisierung


INTERVALLE: dict[str, IntervallInfo] = {
    "Täglich": IntervallInfo("1d", False, None, None, 420, 252.0),
    "Wöchentlich": IntervallInfo("1wk", False, None, None, 1520, 52.0),
    "Intraday 60 Min.": IntervallInfo("1h", True, 60, 729, 45, 252.0 * 8),
    "Intraday 30 Min.": IntervallInfo("30m", True, 30, 59, 21, 252.0 * 16),
    "Intraday 15 Min.": IntervallInfo("15m", True, 15, 59, 11, 252.0 * 32),
    "Intraday 5 Min.": IntervallInfo("5m", True, 5, 59, 5, 252.0 * 96),
}

# Schnellauswahl – beliebig erweiterbar (Börsenkürzel ohne Suffix).
BIST_FAVORITEN: dict[str, str] = {
    "THYAO": "Türk Hava Yolları",
    "GARAN": "Garanti BBVA",
    "AKBNK": "Akbank",
    "ISCTR": "Türkiye İş Bankası (C)",
    "YKBNK": "Yapı ve Kredi Bankası",
    "ASELS": "Aselsan",
    "BIMAS": "BİM Birleşik Mağazalar",
    "KCHOL": "Koç Holding",
    "SAHOL": "Hacı Ömer Sabancı Holding",
    "EREGL": "Ereğli Demir ve Çelik",
    "TUPRS": "Tüpraş",
    "SISE": "Türkiye Şişe ve Cam (Şişecam)",
    "FROTO": "Ford Otosan",
    "TOASO": "Tofaş Türk Otomobil",
    "PGSUS": "Pegasus Hava Taşımacılığı",
    "TCELL": "Turkcell",
    "ENKAI": "Enka İnşaat",
    "ARCLK": "Arçelik",
}
# Alle an der Borsa İstanbul gelisteten Aktien (Kürzel|Name), Stand der TradingView-Übersicht
# „Tüm Türk hisseleri“ (620 Werte). Neue oder umbenannte Werte können hier ergänzt werden.
_BIST_AKTIEN_ROH = """
A1CAP|A1 Capital Yatitim Menkul
A1YEN|A1 Yenilenebilir Enerji Ureti
AAGYO|Agaoglu Avrasya Gayrimen
ACSEL|ACISELSAN ACIPAYAM SE
ADEL|ADEL KALEMCİLİK TİCARE
ADESE|ADESE GAYRİMENKUL YA
ADGYO|Adra Gayrimenkul Yatirim
AEFES|ANADOLU EFES BİRACILI
AFYON|AFYON ÇİMENTO SANAYİ
AGESA|AGESA HAYAT VE EMEKLİ
AGHOL|AG ANADOLU GRUBU HO
AGROT|Agrotech Yuksek Teknoloji
AGYO|ATAKULE GAYRİMENKUL Y
AHGAZ|AHLATCI DOĞAL GAZ DA
AHSGY|Ahes Gayrimenkul Yatirim
AKBNK|AKBANK T.A.Ş
AKCNS|AKÇANSA ÇİMENTO SAN
AKENR|AKENERJİ ELEKTRİK ÜRE
AKFGY|AKFEN GAYRİMENKUL YA
AKFIS|Akfen insaat Turizm ve Tica
AKFYE|AKFEN YENİLENEBİLİR EN
AKGRT|AKSİGORTA A.Ş
AKHAN|Akhan Un Fabrikasi Ve Tari
AKMGY|AKMERKEZ GAYRİMENK
AKSA|AKSA AKRİLİK KİMYA SAN
AKSEN|AKSA ENERJİ ÜRETİM A.Ş
AKSGY|AKİŞ GAYRİMENKUL YATI
AKSUE|AKSU ENERJİ VE TİCARET
AKYHO|AKDENİZ YATIRIM HOLDİ
ALARK|ALARKO HOLDİNG A.Ş
ALBRK|ALBARAKA TÜRK KATILIM
ALBTN|Albayrak Hazir Beton Sana
ALCAR|ALARKO CARRIER SANAYİ
ALCTL|ALCATEL LUCENT TELETA
ALFAS|ALFA SOLAR ENERJİ SAN
ALGYO|ALARKO GAYRİMENKUL Y
ALKA|ALKİM KAĞIT SANAYİ VE T
ALKIM|ALKİM ALKALİ KİMYA A.Ş
ALKLC|Altinkilic Gida ve Sut Sanay
ALTIN|DARPHANE ALTIN SERTİFİ
ALTNY|Altinay Savunma Teknolojil
ALVES|Alves Kablo Sanayi ve Tica
ANELE|ANEL ELEKTRİK PROJE TA
ANGEN|ANATOLİA TANI VE BİYOT
ANHYT|ANADOLU HAYAT EMEKLİ
ANSGR|ANADOLU ANONİM TÜRK
ARASE|DOĞU ARAS ENERJİ YATI
ARCLK|ARÇELİK A.Ş
ARDYZ|ARD GRUP BİLİŞİM TEKN
ARENA|ARENA BİLGİSAYAR SANA
ARFYE|ARF Bio Yenilenebilir Enerji
ARMGD|Armada Gida Ticaret ve S
ARSAN|ARSAN TEKSTİL TİCARET
ARTMS|Artemis Hali A. S
ARZUM|ARZUM ELEKTRİKLİ EV A
ASELS|ASELSAN ELEKTRONİK S
ASGYO|ASCE GAYRIMENKUL YATI
ASTOR|ASTOR ENERJİ A.Ş
ASUZU|ANADOLU ISUZU OTOMO
ATAGY|ATA GAYRİMENKUL YATIRI
ATAKP|Atakey Patates Gida Sanay
ATATP|ATP YAZILIM VE TEKNOLO
ATATR|Ata Turizm Isletmecilik Tasi
ATEKS|AKIN TEKSTİL A.Ş
ATLAS|ATLAS MENKUL KIYMETL
ATSYH|ATLANTİS YATIRIM HOLDİ
AVGYO|AVRASYA GAYRİMENKUL
AVHOL|AVRUPA YATIRIM HOLDİN
AVOD|A.V.O.D. KURUTULMUŞ GI
AVPGY|Avrupakent Gayrimenkul Y
AYCES|ALTIN YUNUS ÇEŞME TUR
AYDEM|AYDEM YENİLENEBİLİR E
AYEN|AYEN ENERJİ A.Ş
AYES|AYES ÇELİK HASIR VE ÇİT
AYGAZ|AYGAZ A.Ş
AZTEK|AZTEK TEKNOLOJİ ÜRÜN
BAGFS|BAGFAŞ BANDIRMA GÜB
BAHKM|Bahadir Kimya Sanayi Ve
BAKAB|BAK AMBALAJ SANAYİ V
BALAT|BALATACILAR BALATACILI
BALSU|Balsu Gida Sanayi ve Ticar
BANVT|BANVİT BANDIRMA VİTA
BARMA|BAREM AMBALAJ SANAY
BASCM|BAŞTAŞ BAŞKENT ÇİMEN
BASGZ|BAŞKENT DOĞALGAZ DA
BAYRK|BAYRAK EBT TABAN SANA
BEGYO|Bati Ege Gayrimenkul Yatiri
BERA|BERA HOLDİNG A.Ş
BESLR|Besler Gida Ve Kimya Sana
BESTE|Best Brands Grup Enerji Ya
BETAE|Beta Enerji ve Teknoloji AS
BEYAZ|BEYAZ FİLO OTO KİRALA
BFREN|BOSCH FREN SİSTEMLERİ
BIENY|BİEN YAPI ÜRÜNLERİ SAN
BIGCH|BÜYÜK ŞEFLER GIDA TURİ
BIGEN|Birlesim Grup Enerji Yatiriml
BIGTK|Big Medya Teknoloji A.S
BIMAS|BİM BİRLEŞİK MAĞAZALA
BINBN|Bin Ulasim Ve Akilli Sehir Te
BINHO|1000 Yatirimlar Holding AS
BIOEN|BİOTREND ÇEVRE VE ENE
BIZIM|BİZİM TOPTAN SATIŞ MAĞ
BJKAS|BEŞİKTAŞ FUTBOL YATIRI
BKRGY|Bakirci Gayrimenkul Yatirim
BLCYT|BİLİCİ YATIRIM SANAYİ VE
BLUME|Blume Metal Kimya Anoni
BMSCH|BMS ÇELİK HASIR SANAY
BMSTL|BMS BİRLEŞİK METAL SA
BNTAS|BANTAŞ BANDIRMA AMB
BOBET|BOĞAZİÇİ BETON SANAYİ
BORLS|Borlease Otomotiv AS
BORSK|Bor Seker A.S
BOSSA|BOSSA TİCARET VE SANA
BRISA|BRİSA BRIDGESTONE SAB
BRKO|BİRKO BİRLEŞİK KOYUNLU
BRKSN|BERKOSAN YALITIM VE T
BRKVY|BİRİKİM VARLIK YÖNETİM
BRLSM|BİRLEŞİM MÜHENDİSLİK I
BRMEN|BİRLİK MENSUCAT TİCAR
BRSAN|BORUSAN MANNESMAN
BRYAT|BORUSAN YATIRIM VE PA
BSOKE|BATISÖKE SÖKE ÇİMENT
BTCIM|BATIÇİM BATI ANADOLU Ç
BUCIM|BURSA ÇİMENTO FABRİK
BULGS|Bulls Girisim Sermayesi Yat
BURCE|BURÇELİK BURSA ÇELİK
BURVA|BURÇELİK VANA SANAYİ V
BVSAN|BÜLBÜLOĞLU VİNÇ SANA
BYDNR|Baydoner Restoranlari A.S
CANTE|ÇAN2 TERMİK A.Ş
CASA|CASA EMTİA PETROL KİM
CATES|Cates Elektrik Uretim Anoni
CCOLA|COCA-COLA İÇECEK A.Ş
CELHA|ÇELİK HALAT VE TEL SAN
CEMAS|ÇEMAŞ DÖKÜM SANAYİ
CEMTS|ÇEMTAŞ ÇELİK MAKİNA S
CEMZY|CEM ZEYTIN ANONIM SIR
CEOEM|CEO EVENT MEDYA A.Ş
CGCAM|Cagdas Cam Sanayi ve Ti
CIMSA|ÇİMSA ÇİMENTO SANAYİ
CITAS|Citlekci Magazacilik Gida AS
CLEBI|ÇELEBİ HAVA SERVİSİ A.Ş
CMBTN|ÇİMBETON HAZIRBETON
CMENT|ÇİMENTAŞ İZMİR ÇİMENT
CONSE|CONSUS ENERJİ İŞLETM
COSMO|COSMOS YATIRIM HOLDİ
CRDFA|CREDITWEST FAKTORİNG
CRFSA|CARREFOURSA CARREFO
CUSAN|ÇUHADAROĞLU METAL S
CVKMD|CVK MADEN İŞLETMELE
CWENE|CW ENERJİ MÜHENDİSLİ
DAGI|DAGİ GİYİM SANAYİ VE Tİ
DAPGM|DAP GAYRİMENKUL GELİ
DARDL|DARDANEL ÖNENTAŞ GID
DCTTR|DCT Trading Dis Ticaret An
DENGE|DENGE YATIRIM HOLDİNG
DERHL|DERLÜKS YATIRIM HOLDİ
DERIM|DERİMOD KONFEKSİYON
DESA|DESA DERİ SANAYİ VE TİC
DESPC|DESPEC BİLGİSAYAR PAZ
DEVA|DEVA HOLDİNG A.Ş
DGATE|DATAGATE BİLGİSAYAR M
DGGYO|DOĞUŞ GAYRİMENKUL Y
DGNMO|DOĞANLAR MOBİLYA GR
DIRIT|DİRİTEKS DİRİLİŞ TEKSTİL
DITAS|DİTAŞ DOĞAN YEDEK PAR
DMLKT|Emlak Konut Gayrimenkul
DMRGD|DMR Unlu Mamuller Ureti
DMSAS|DEMİSAŞ DÖKÜM EMAYE
DNISI|DİNAMİK ISI MAKİNA YALI
DOAS|DOĞUŞ OTOMOTİV SERVİ
DOFER|Dofer Yapi Maizemeleri Sa
DOFRB|DOF Robotik Sanayi Anoni
DOGUB|DOĞUSAN BORU SANAYİİ
DOHOL|DOĞAN ŞİRKETLER GRU
DOKTA|DÖKTAŞ DÖKÜMCÜLÜK T
DSTKF|DESTEK FAKTORİNG A.Ş
DUNYH|Dunya Holding Anonim Sir
DURDO|DURAN DOĞAN BASIM V
DURKN|Durukan Sekerleme Sanayi
DYOBY|DYO BOYA FABRİKALARI S
DZGYO|DENİZ GAYRİMENKUL YAT
EBEBK|EBEBEK MAGAZACILIK AN
ECILC|EİS ECZACIBAŞI İLAÇ SIN
ECOGR|Ecogreen Enerji Holding A.S
ECZYT|ECZACIBAŞI YATIRIM HOL
EDATA|E-DATA TEKNOLOJİ PAZA
EDIP|EDİP GAYRİMENKUL YATIR
EFOR|Efor Yatirim Sanayi Ticaret
EGEEN|EGE ENDÜSTRİ VE TİCAR
EGEGY|Egeyapi Avrupa Gayrimenk
EGEPO|NASMED ÖZEL SAĞLIK Hİ
EGGUB|EGE GÜBRE SANAYİİ A.Ş
EGPRO|EGE PROFİL TİCARET VE
EGSER|EGE SERAMİK SANAYİ VE
EKDMR|Ekinciler Demir ve Celik Sa
EKIM|EKİM TURİZM TİCARET VE
EKIZ|EKİZ KİMYA SANAYİ VE Tİ
EKOS|Ekos Teknoloji ve Elektrik AS
EKSUN|EKSUN GIDA TARIM SANA
ELITE|ELİTE NATUREL ORGANİK
EMKEL|EMEK ELEKTRİK ENDÜST
EMNIS|EMİNİŞ AMBALAJ SANAYİ
EMPAE|Empa Elektronik Sanayi ve
ENDAE|Enda Enerji Holding Anoni
ENERY|Enerya Enerji A.S
ENJSA|ENERJİSA ENERJİ A.Ş
ENKAI|ENKA İNŞAAT VE SANAYİ
ENPRA|Enpara Bank A.S
ENSRI|ENSARİ DERİ GIDA SANAYİ
ENTRA|IC Enterra Yenilenebilir Ene
EPLAS|EGEPLAST EGE PLASTİK
ERBOS|ERBOSAN ERCİYAS BORU
ERCB|ERCİYAS ÇELİK BORU SA
EREGL|EREĞLİ DEMİR VE ÇELİK F
ERSU|ERSU MEYVE VE GIDA SA
ESCAR|ESCAR FİLO KİRALAMA Hİ
ESCOM|ESCORT TEKNOLOJİ YATI
ESEN|ESENBOĞA ELEKTRİK ÜR
ETILR|ETİLER GIDA VE TİCARİ YA
ETYAT|EURO TREND YATIRIM OR
EUHOL|EURO YATIRIM HOLDİNG
EUKYO|EURO KAPİTAL YATIRIM O
EUPWR|EUROPOWER ENERJİ VE
EUREN|EUROPEN ENDÜSTRİ İNŞ
EUYO|EURO MENKUL KIYMET YA
EYGYO|EYG GAYRİMENKUL YATIRI
FADE|FADE GIDA YATIRIM SANA
FENER|FENERBAHÇE FUTBOL A.Ş
FLAP|FLAP KONGRE TOPLANTI
FMIZP|FEDERAL-MOGUL İZMİT P
FONET|FONET BİLGİ TEKNOLOJİL
FORMT|FORMET METAL VE CAM
FORTE|FORTE BILGI ILETISIM TEK
FRIGO|FRİGO-PAK GIDA MADDE
FRMPL|Formul Plastik Ve Metal Sa
FROTO|FORD OTOMOTİV SANAYİ
FZLGY|FUZUL GAYRIMENKUL YAT
GARAN|TÜRKİYE GARANTİ BANK
GARFA|GARANTİ FAKTORİNG A.Ş
GATEG|Gate Group Teknoloji Medy
GEDIK|GEDİK YATIRIM MENKUL D
GEDZA|GEDİZ AMBALAJ SANAYİ
GENIL|GEN İLAÇ VE SAĞLIK ÜRÜ
GENKM|Gentas Kimya Sanayi ve Ti
GENTS|GENTAŞ DEKORATİF YÜZ
GEREL|GERSAN ELEKTRİK TİCAR
GESAN|GİRİŞİM ELEKTRİK SANAY
GIPTA|Gipta Ofis Kirtasiye ve Pro
GLBMD|GLOBAL MENKUL DEĞER
GLCVY|GELECEK VARLIK YÖNETİ
GLRMK|Gulermak Agir Sanayi Insa
GLRYH|GÜLER YATIRIM HOLDİNG
GLYHO|GLOBAL YATIRIM HOLDİN
GMTAS|GİMAT MAĞAZACILIK SA
GOKNR|GÖKNUR GIDA MADDELE
GOLDA|Golda Gida Sanayi ve Ticar
GOLTS|GÖLTAŞ GÖLLER BÖLGES
GOODY|GOODYEAR LASTİKLERİ T
GOZDE|GÖZDE GİRİŞİM SERMAY
GRNYO|GARANTİ YATIRIM ORTAK
GRSEL|GÜR-SEL TURİZM TAŞIMA
GRTHO|Grainturk Holding A.S
GSDDE|GSD DENİZCİLİK GAYRİM
GSDHO|GSD HOLDİNG A.Ş
GSRAY|GALATASARAY SPORTİF S
GUBRF|GÜBRE FABRİKALARI T.A.Ş
GUNDG|Gundogdu Gida Sut Urunl
GWIND|GALATA WIND ENERJİ A.Ş
GZNMI|GEZİNOMİ SEYAHAT TURİ
HALKB|TÜRKİYE HALK BANKASI
HATEK|HATEKS HATAY TEKSTİL İ
HATSN|Hat-San Gemi Insaa Bakim
HDFGS|HEDEF GİRİŞİM SERMAYE
HEDEF|HEDEF HOLDİNG A.Ş
HEKTS|HEKTAŞ TİCARET T.A.Ş
HKTM|HİDROPAR HAREKET KON
HLGYO|HALK GAYRİMENKUL YATI
HOROZ|Horoz Lojistik Kargo Hizme
HRKET|Hareket Proje Tasimaciligi v
HTTBT|HİTİT BİLGİSAYAR HİZMET
HUBVC|HUB GİRİŞİM SERMAYESİ
HUNER|HUN YENİLENEBİLİR ENE
HURGZ|HÜRRİYET GAZETECİLİK
ICBCT|ICBC TURKEY BANK A.Ş
ICUGS|ICU Girisim Sermayesi Yatir
IDGYO|İDEALİST GAYRİMENKUL Y
IEYHO|IŞIKLAR ENERJİ VE YAPI H
IHAAS|İHLAS HABER AJANSI A.Ş
IHEVA|İHLAS EV ALETLERİ İMAL
IHGZT|İHLAS GAZETECİLİK A.Ş
IHLAS|İHLAS HOLDİNG A.Ş
IHLGM|İHLAS GAYRİMENKUL PR
IHYAY|İHLAS YAYIN HOLDİNG A.Ş
IMASM|İMAŞ MAKİNA SANAYİ A.Ş
INDES|İNDEKS BİLGİSAYAR SİSTE
INFO|İNFO YATIRIM MENKUL DE
INGRM|INGRAM MİCRO BİLİŞİM Sİ
INTEK|Innosa Teknoloji Anonim Sir
INTEM|İNTEMA İNŞAAT VE TESİS
INTET|Intetra Teknoloji ve Bilisim
INVEO|INVEO YATIRIM HOLDİNG
INVES|INVESTCO HOLDİNG A.Ş
ISBIR|İŞBİR HOLDİNG A.Ş
ISDMR|İSKENDERUN DEMİR VE Ç
ISFIN|İŞ FİNANSAL KİRALAMA A
ISGYO|İŞ GAYRİMENKUL YATIRIM
ISKPL|IŞIK PLASTİK SANAYİ VE D
ISMEN|İŞ YATIRIM MENKUL DEĞE
ISSEN|İŞBİR SENTETİK DOKUMA
ISVEA|Isvea Seramik ve Banyo Ur
IZENR|Izdemir Enerji Elektrik Ureti
IZFAS|İZMİR FIRÇA SANAYİ VE Tİ
IZINV|İZ YATIRIM HOLDİNG A.Ş
IZMDC|İZMİR DEMİR ÇELİK SANA
JANTS|JANTSA JANT SANAYİ VE
KAPLM|KAPLAMİN AMBALAJ SA
KARCL|Kardemir Celik Sanayi AS
KAREL|KAREL ELEKTRONİK SAN
KARSN|KARSAN OTOMOTİV SAN
KARTN|KARTONSAN KARTON SA
KATMR|KATMERCİLER ARAÇ ÜST
KAYSE|KAYSERİ ŞEKER FABRİKA
KBORU|Kuzey Boru A.S
KCAER|KOCAER ÇELİK SANAYİ VE
KCHOL|KOÇ HOLDİNG A.Ş
KENT|KENT GIDA MADDELERİ S
KERVN|KERVANSARAY YATIRIM H
KFEIN|KAFEİN YAZILIM HİZMETL
KGYO|KORAY GAYRİMENKUL YA
KIMMR|ERSAN ALIŞVERİŞ HİZME
KLGYO|KİLER GAYRİMENKUL YATI
KLKIM|KALEKİM KİMYEVİ MADD
KLMSN|KLİMASAN KLİMA SANAYİ
KLNMA|TÜRKİYE KALKINMA VE Y
KLRHO|KİLER HOLDİNG A.Ş
KLSER|Kaleseramik Canakkale Kal
KLSYN|KOLEKSİYON MOBİLYA SA
KLYPV|Kalyon Gunes Teknolojileri
KMPUR|KİMTEKS POLİÜRETAN S
KNFRT|KONFRUT GIDA SANAYİ V
KOCMT|Koc Metalurji AS
KONKA|KONYA KAĞIT SANAYİ VE
KONTR|KONTROLMATİK TEKNOL
KONYA|KONYA ÇİMENTO SANAYİİ
KOPOL|KOZA POLYESTER SANAYİ
KORDS|KORDSA TEKNİK TEKSTİL
KOTON|KOTON MAĞAZACILIK TE
KPEKS|Kapeks Kimya Sanayi AS
KRDMA|KARDEMİR KARABÜK DE
KRGYO|KÖRFEZ GAYRİMENKUL Y
KRONT|KRON TELEKOMÜNİKASY
KRPLS|KOROPLAST TEMİZLİK AM
KRSTL|KRİSTAL KOLA VE MEŞRU
KRTEK|KARSU TEKSTİL SANAYİİ
KRVGD|KERVAN GIDA SANAYİ VE
KSTUR|KUŞTUR KUŞADASI TURİZ
KTLEV|KATILIMEVIM TASARRUF
KTSKR|KÜTAHYA ŞEKER FABRİKA
KUTPO|KÜTAHYA PORSELEN SA
KUVVA|KUVVA GIDA TİCARET VE
KUYAS|KUYAŞ YATIRIM A.Ş
KZBGY|KIZILBÜK GAYRİMENKUL
KZGYO|Kuzugrup Gayrimenkul Yati
LIDER|LDR TURİZM A.Ş
LIDFA|LİDER FAKTORİNG A.Ş
LILAK|Lila Kagit Sanayi Ve Ticaret
LINK|LİNK BİLGİSAYAR SİSTEML
LKMNH|LOKMAN HEKİM ENGÜRÜ
LMKDC|Limak Dogu Anadolu Cime
LOGO|LOGO YAZILIM SANAYİ VE
LRSHO|Loras Holding Anonim Sirk
LUKSK|LÜKS KADİFE TİCARET VE
LXGYO|Luxera Gayrimenkul Yatirim
LYDHO|Lydia Holding A.S
LYDYE|Lydia Yesil Enerji kaynaklari
MAALT|MARMARİS ALTINYUNUS
MACKO|MACKOLİK İNTERNET HİZ
MAGEN|MARGÜN ENERJİ ÜRETİM
MAKIM|MAKİM MAKİNA TEKNOL
MAKTK|MAKİNA TAKIM ENDÜSTR
MANAS|MANAS ENERJİ YÖNETİM
MARBL|Tureks Turunc Madencilik I
MARMR|Marmara Holding AS
MARTI|MARTI OTEL İŞLETMELERİ
MASFN|Masfen Enerji AS
MAVI|MAVİ GİYİM SANAYİ VE Tİ
MCARD|Metropal Kurumsal Hizmet
MEDTR|MEDİTERA TIBBİ MALZEM
MEGAP|MEGA POLİETİLEN KÖPÜ
MEGMT|Mega Metal Sanayi Ve Tic
MEKAG|Meka Global Makine Imala
MEPET|MEPET METRO PETROL V
MERCN|MERCAN KİMYA SANAYİ V
MERIT|MERİT TURİZM YATIRIM V
MERKO|MERKO GIDA SANAYİ VE
METEN|METGUN Enerji Yatirimlari
METRO|METRO TİCARİ VE MALİ Y
MEYSU|Meysu Gida Sanayi Ve Tic
MGROS|MİGROS TİCARET A.Ş
MHRGY|MHR Gayrimenkul Yatirim
MIATK|MİA TEKNOLOJİ A.Ş
MMCAS|MMC SANAYİ VE TİCARİ Y
MNDRS|MENDERES TEKSTİL SAN
MNDTR|MONDİ TURKEY OLUKLU
MOBTL|MOBİLTEL İLETİŞİM HİZM
MOGAN|Mogan Enerji Yatirim Holdi
MOPAS|Mopas Marketcilik Gida Sa
MPARK|MLP SAĞLIK HİZMETLERİ
MRGYO|MARTI GAYRİMENKUL YA
MRSHL|MARSHALL BOYA VE VER
MSGYO|MİSTRAL GAYRİMENKUL
MTRKS|MATRİKS FİNANSAL TEK
MTRYO|METRO YATIRIM ORTAKLI
MZHLD|MAZHAR ZORLU HOLDİN
NATEN|NATUREL YENİLENEBİLİR
NETAS|NETAŞ TELEKOMÜNİKASY
NETCD|Netcad Yazilim A.S
NETGL|Net Global Endustriyel Yatir
NIBAS|NİĞBAŞ NİĞDE BETON SA
NTGAZ|NATURELGAZ SANAYİ VE
NTHOL|NET HOLDİNG A.Ş
NUGYO|NUROL GAYRİMENKUL YA
NUHCM|NUH ÇİMENTO SANAYİ A
OBAMS|Oba Makarnacilik Sanayi V
OBASE|OBASE BİLGİSAYAR VE DA
ODAS|ODAŞ ELEKTRİK ÜRETİM
ODINE|Odine Solutions Teknoloji T
OFSYM|Ofis Yem Gida Sanayi ve Ti
ONCSM|ONCOSEM ONKOLOJİK S
ONRYT|Onur Yuksek Teknoloji AS
ORCAY|ORÇAY ORTAKÖY ÇAY SA
ORGE|ORGE ENERJİ ELEKTRİK T
ORMA|ORMA ORMAN MAHSULL
ORZAX|Orzaks Ilac ve Kimya Sana
OSMEN|OSMANLI YATIRIM MENK
OSTIM|OSTİM ENDÜSTRİYEL YATI
OTKAR|OTOKAR OTOMOTİV VE S
OTTO|OTTO HOLDİNG A.Ş
OYAKC|OYAK ÇİMENTO FABRİKAL
OYAYO|OYAK YATIRIM ORTAKLIĞI
OYLUM|OYLUM SINAİ YATIRIMLA
OYYAT|OYAK YATIRIM MENKUL D
OZATD|OZATA DENIZCILIK SANAY
OZGYO|ÖZDERİCİ GAYRİMENKUL
OZKGY|ÖZAK GAYRİMENKUL YATI
OZRDN|ÖZERDEN PLASTİK SANA
OZSUB|ÖZSU BALIK ÜRETİM A.Ş
OZYSR|Ozyasar Tel ve Galvanizlem
PAGYO|PANORA GAYRİMENKUL Y
PAHOL|PASIFIK HOLDING A.S
PAMEL|PAMEL YENİLENEBİLİR EL
PAPIL|PAPİLON SAVUNMA TEKN
PARSN|PARSAN MAKİNA PARÇAL
PASEU|Pasifik Eurasia Lojistik dis T
PATEK|Pasifik Teknoloji AS
PCILT|PC İLETİŞİM VE MEDYA Hİ
PEKGY|PEKER GAYRİMENKUL YA
PENGD|PENGUEN GIDA SANAYİ A
PENTA|PENTA TEKNOLOJİ ÜRÜN
PETKM|PETKİM PETROKİMYA HO
PETUN|PINAR ENTEGRE ET VE U
PGSUS|PEGASUS HAVA TAŞIMACI
PINSU|PINAR SU VE İÇECEK SAN
PKART|PLASTİKKART AKILLI KAR
PKENT|PETROKENT TURİZM A.Ş
PLTUR|PLATFORM TURİZM TAŞI
PNLSN|PANELSAN ÇATI CEPHE Sİ
PNSUT|PINAR SÜT MAMULLERİ S
POLHO|POLİSAN HOLDİNG A.Ş
POLTK|POLİTEKNİK METAL SANA
PRDGS|PARDUS GİRİŞİM SERMAY
PRKAB|TÜRK PRYSMİAN KABLO
PRKME|PARK ELEKTRİK ÜRETİM
PRZMA|PRİZMA PRES MATBAACI
PSDTC|PERGAMON STATUS DIŞ T
PSGYO|PASİFİK GAYRİMENKUL YA
QNBFK|QNB Finansal Kiralama A.S
QNBTR|QNB Bank AS
QUAGR|QUA GRANITE HAYAL YAP
QUICK|Quick Sigorta AS
RALYH|RAL YATIRIM HOLDİNG A.Ş
RAYSG|RAY SİGORTA A.Ş
REEDR|Reeder Teknoloji Sanayi ve
RGYAS|RÖNESANS GAYRİMENKU
RNPOL|RAİNBOW POLİKARBONA
RODRG|RODRİGO TEKSTİL SANA
RTALB|RTA LABORATUVARLARI
RUBNS|RUBENİS TEKSTİL SANAYİ
RUZYE|Ruzy Madencilik Ve Enerji Y
RYGYO|REYSAŞ GAYRİMENKUL Y
RYSAS|REYSAŞ TAŞIMACILIK VE L
SAFKR|SAFKAR EGE SOĞUTMAC
SAHOL|HACI ÖMER SABANCI HO
SAMAT|SARAY MATBAACILIK KA
SANEL|SAN-EL MÜHENDİSLİK EL
SANFM|SANİFOAM ENDÜSTRİ VE
SANKO|SANKO PAZARLAMA İTH
SARAE|SA-RA Enerji Insaat Ticaret
SARKY|SARKUYSAN ELEKTROLİT
SASA|SASA POLYESTER SANAYİ
SAYAS|SAY YENİLENEBİLİR ENER
SDTTR|SDT UZAY VE SAVUNMA T
SEGMN|Segmen Kardesler Gida Ur
SEGYO|ŞEKER GAYRİMENKUL YA
SEKFK|ŞEKER FİNANSAL KİRALA
SEKUR|SEKURO PLASTİK AMBAL
SELEC|SELÇUK ECZA DEPOSU Tİ
SELVA|SELVA GIDA SANAYİ A.Ş
SERNT|Seranit Granit Seramik San
SEYKM|SEYİTLER KİMYA SANAYİ
SILVR|SİLVERLİNE ENDÜSTRİ VE
SISE|TÜRKİYE ŞİŞE VE CAM FA
SKBNK|ŞEKERBANK T.A.Ş
SKTAS|SÖKTAŞ TEKSTİL SANAYİ
SKYLP|Skyalp Finansal Teknolojiler
SKYMD|Seker Yatirim Menkul Dege
SMART|SMARTİKS YAZILIM A.Ş
SMRTG|SMART GÜNEŞ ENERJİSİ
SMRVA|Sumer Varlik Yonetim A.S
SNGYO|SİNPAŞ GAYRİMENKUL YA
SNICA|SANİCA ISI SANAYİ A.Ş
SNPAM|SÖNMEZ PAMUKLU SANA
SODSN|SODAŞ SODYUM SANAYİİ
SOHOE|Soho Giyim ve Enerji A.S
SOKE|SÖKE DEĞİRMENCİLİK SA
SOKM|ŞOK MARKETLER TİCARE
SONME|SÖNMEZ FİLAMENT SEN
SRVGY|SERVET GAYRİMENKUL Y
SSAAT|Saat ve Saat Sanayi ve Tic
SUMAS|SUMAŞ SUNİ TAHTA VE M
SUNTK|SUN TEKSTİL SANAYİ VE
SURGY|Sur Tatil Evleri Gayrimenkul
SUWEN|SUWEN TEKSTİL SANAYİ
SVGYO|Savur Gayrimenkul Yatirim
TABGD|TAB Gida Sanayi ve Ticaret
TARKM|Tarkim Bitki Koruma Sanay
TATEN|Tatlipinar Enerji Uretim A.S
TATGD|TAT GIDA SANAYİ A.Ş
TAVHL|TAV HAVALİMANLARI HOL
TBORG|TÜRK TUBORG BİRA VE M
TCELL|TURKCELL İLETİŞİM HİZM
TCKRC|Kirac Galvaniz Telekominik
TDGYO|TREND GAYRİMENKUL YA
TEHOL|Tera Yatirim Teknoloji Holdi
TEKTU|TEK-ART İNŞAAT TİCARET
TERA|TERA YATIRIM MENKUL D
TEZOL|EUROPAP TEZOL KAĞIT S
TGSAS|TGS DIŞ TİCARET A.Ş
THYAO|TÜRK HAVA YOLLARI A.O
TKFEN|TEKFEN HOLDİNG A.Ş
TKNKA|Teknika Plast Teknik Kalip
TKNSA|TEKNOSA İÇ VE DIŞ TİCA
TLMAN|TRABZON LİMAN İŞLETM
TMPOL|TEMAPOL POLİMER PLAS
TMSN|TÜMOSAN MOTOR VE TR
TNZTP|TAPDİ OKSİJEN ÖZEL SAĞ
TOASO|TOFAŞ TÜRK OTOMOBİL F
TRALT|Turk Altin Isletmeleri A.S
TRCAS|TURCAS PETROL A.Ş
TRENJ|TR Dogal Enerji Kaynaklari
TRGYO|TORUNLAR GAYRİMENKU
TRHOL|Tera Financial Investments
TRILC|TURK İLAÇ VE SERUM SA
TRMET|TR Anadolu Metal Madenc
TSGYO|TSKB GAYRİMENKUL YATI
TSKB|TÜRKİYE SINAİ KALKINMA
TSPOR|TRABZONSPOR SPORTİF
TTKOM|TÜRK TELEKOMÜNİKASY
TTRAK|TÜRK TRAKTÖR VE ZİRAA
TUCLK|TUĞÇELİK ALÜMİNYUM V
TUKAS|TUKAŞ GIDA SANAYİ VE Tİ
TUPRS|TÜPRAŞ-TÜRKİYE PETRO
TUREX|TUREKS TURİZM TAŞIMA
TURGG|TÜRKER PROJE GAYRİME
TURSG|TÜRKİYE SİGORTA A.Ş
UCAYM|Ucay Muhendislik Enerji ve
UFUK|UFUK YATIRIM YÖNETİM V
ULAS|ULAŞLAR TURİZM YATIRI
ULKER|ÜLKER BİSKÜVİ SANAYİ A
ULUFA|ULUSAL FAKTORİNG A.Ş
ULUSE|ULUSOY ELEKTRİK İMALA
ULUUN|ULUSOY UN SANAYİ VE Tİ
UNLU|ÜNLÜ YATIRIM HOLDİNG A
USAK|UŞAK SERAMİK SANAYİ A.Ş
USHOL|US Yatirim Holding A.S
VAKBN|TÜRKİYE VAKIFLAR BANK
VAKFA|VAKIF FAKTORİNG A.Ş
VAKFN|VAKIF FİNANSAL KİRALA
VAKKO|VAKKO TEKSTİL VE HAZIR
VANGD|VANET GIDA SANAYİ İÇ V
VBTYZ|VBT YAZILIM A.Ş
VERUS|VERUSA HOLDİNG A.Ş
VESBE|VESTEL BEYAZ EŞYA SAN
VESTL|VESTEL ELEKTRONİK SAN
VEYAS|Turker Vangolu Enerji Yatiri
VKFYO|VAKIF MENKUL KIYMET Y
VKGYO|VAKIF GAYRİMENKUL YATI
VKING|VİKİNG KAĞIT VE SELÜLO
VRGYO|Vera Konsept Gayrimenkul
VSNMD|Visne Madencilik Uretim S
YAPRK|YAPRAK SÜT VE BESİ ÇİF
YATAS|YATAŞ YATAK VE YORGAN
YAYLA|YAYLA ENERJİ ÜRETİM TU
YBTAS|YİBİTAŞ YOZGAT İŞÇİ BİRL
YEOTK|YEO TEKNOLOJİ ENERJİ V
YESIL|YEŞİL YATIRIM HOLDİNG A
YGGYO|YENİ GİMAT GAYRİMENKU
YIGIT|Yigit Aku Malzemeleri Nakli
YKBNK|YAPI VE KREDİ BANKASI
YKSLN|YÜKSELEN ÇELİK A.Ş
YONGA|YONGA MOBİLYA SANAYİ
YUNSA|YÜNSA YÜNLÜ SANAYİ VE
YYAPI|YEŞİL YAPI ENDÜSTRİSİ A.Ş
YYLGD|YAYLA AGRO GIDA SANAY
ZEDUR|ZEDUR ENERJİ ELEKTRİK
ZERGY|Zeray Gayrimenkul Yatirim
ZGYO|Z Gayrimenkul Yatirim Orta
ZOREN|ZORLU ENERJİ ELEKTRİK
ZRGYO|ZİRAAT GAYRİMENKUL YA
"""
BIST_ALLE_AKTIEN: dict[str, str] = {
    zeile.split("|", 1)[0].strip(): zeile.split("|", 1)[1].strip()
    for zeile in _BIST_AKTIEN_ROH.strip().splitlines() if "|" in zeile
}

BIST_INDIZES: dict[str, str] = {
    "XU100": "BIST 100",
    "XU030": "BIST 30",
    "XU050": "BIST 50",
    "XBANK": "BIST Banken",
    "XUSIN": "BIST Industrie",
    "XHOLD": "BIST Holdings",
    "XUTEK": "BIST Technologie",
}
INDEX_ALIASE: dict[str, str] = {
    "BIST100": "XU100", "BIST30": "XU030", "BIST50": "XU050",
    "BISTBANKA": "XBANK", "BISTBANK": "XBANK", "BISTSINAI": "XUSIN", "BISTHOLDING": "XHOLD",
}

OVERLAY_OPTIONEN = ["SMA 20", "SMA 50", "SMA 200", "EMA 12", "EMA 26", "Bollinger-Bänder",
                    "Unterstützung/Widerstand", "Signale"]
OVERLAY_STANDARD = ["SMA 20", "SMA 50", "SMA 200", "Bollinger-Bänder", "Unterstützung/Widerstand", "Signale"]
PANEL_OPTIONEN = ["Volumen", "RSI", "MACD", "Stochastik", "ADX", "OBV", "ATR"]
PANEL_STANDARD = ["Volumen", "RSI", "MACD"]

# Farben (in hellen und dunklen Ansichten gut lesbar)
FARBEN: dict[str, str] = {
    "positiv": "#16a34a",
    "negativ": "#dc2626",
    "neutral": "#d97706",
    "kerze_hoch": "#16a34a",
    "kerze_tief": "#dc2626",
    "sma20": "#3b82f6",
    "sma50": "#f59e0b",
    "sma200": "#a855f7",
    "ema12": "#06b6d4",
    "ema26": "#ec4899",
    "bb": "#94a3b8",
    "linie1": "#3b82f6",
    "linie2": "#f59e0b",
    "grau": "#94a3b8",
}
STIL_FARBEN: dict[str, str] = {
    "stark_pos": "#15803d", "pos": "#16a34a", "neutral": "#d97706", "neg": "#dc2626", "stark_neg": "#b91c1c",
}

# Signaltypen
KAUF_STARK = "Starkes Kaufsignal"
KAUF = "Normales Kaufsignal"
HALTEN = "Halten/Beobachten"
VERKAUF = "Normales Verkaufssignal"
VERKAUF_STARK = "Starkes Verkaufssignal"
SIGNAL_CODES: dict[int, str] = {2: KAUF_STARK, 1: KAUF, 0: HALTEN, -1: VERKAUF, -2: VERKAUF_STARK}
SIGNAL_STIL: dict[str, str] = {KAUF_STARK: "stark_pos", KAUF: "pos", HALTEN: "neutral",
                               VERKAUF: "neg", VERKAUF_STARK: "stark_neg"}

PUNKTE_SPALTEN: dict[str, str] = {
    "P_SMA_Kreuz": "Golden/Death Cross (SMA 50/200)",
    "P_RSI": "RSI 14",
    "P_MACD": "MACD-Kreuzung",
    "P_SMA200": "Kurs vs. SMA 200",
    "P_Volumen": "Volumen-Signal",
    "P_ADX": "ADX-Trendbestätigung",
}


# =============================================================================
# 1) Parameter (Datenklassen)
# =============================================================================

@dataclass
class PunkteGewichte:
    """Punkte je Bewertungskomponente (in der Oberfläche anpassbar)."""

    golden_cross: int = 2
    death_cross: int = -2
    rsi_ueberverkauft: int = 2      # RSI unter der Überverkauft-Schwelle (Standard 30)
    rsi_30_bis_50: int = 1          # RSI zwischen Überverkauft-Schwelle und 50
    rsi_ueberkauft: int = -2        # RSI über der Überkauft-Schwelle (Standard 70)
    macd_kreuz_hoch: int = 2
    macd_kreuz_runter: int = -2
    kurs_ueber_sma200: int = 1
    kurs_unter_sma200: int = -1
    volumen_positiv: int = 1
    volumen_negativ: int = -1
    adx_trend_positiv: int = 1
    adx_trend_negativ: int = -1

    def _gruppen(self) -> list[list[int]]:
        return [
            [self.golden_cross, self.death_cross, 0],
            [self.rsi_ueberverkauft, self.rsi_30_bis_50, self.rsi_ueberkauft, 0],
            [self.macd_kreuz_hoch, self.macd_kreuz_runter, 0],
            [self.kurs_ueber_sma200, self.kurs_unter_sma200, 0],
            [self.volumen_positiv, self.volumen_negativ, 0],
            [self.adx_trend_positiv, self.adx_trend_negativ, 0],
        ]

    def maximum(self) -> int:
        """Höchstmögliche Gesamtpunktzahl."""
        return int(sum(max(g) for g in self._gruppen()))

    def minimum(self) -> int:
        """Niedrigstmögliche Gesamtpunktzahl."""
        return int(sum(min(g) for g in self._gruppen()))


GEWICHT_NAMEN: dict[str, str] = {
    "golden_cross": "Golden Cross",
    "death_cross": "Death Cross",
    "rsi_ueberverkauft": "RSI überverkauft",
    "rsi_30_bis_50": "RSI überverkauft bis 50",
    "rsi_ueberkauft": "RSI überkauft",
    "macd_kreuz_hoch": "MACD-Kreuz nach oben",
    "macd_kreuz_runter": "MACD-Kreuz nach unten",
    "kurs_ueber_sma200": "Kurs über SMA 200",
    "kurs_unter_sma200": "Kurs unter SMA 200",
    "volumen_positiv": "Volumen-Signal positiv",
    "volumen_negativ": "Volumen-Signal negativ",
    "adx_trend_positiv": "ADX stark, Trend aufwärts",
    "adx_trend_negativ": "ADX stark, Trend abwärts",
}


@dataclass
class StrategieParameter:
    """Parameter für Punktesystem und Signalerzeugung."""

    rsi_ueberverkauft: float = 30.0
    rsi_ueberkauft: float = 70.0
    rsi_kauf_obergrenze: float = 65.0     # RSI-Obergrenze für Kaufsignale
    adx_schwelle: float = 25.0
    volumen_faktor: float = 1.2           # Volumen gilt ab Faktor × 20-Perioden-Durchschnitt als erhöht
    macd_fenster: int = 3                 # MACD-Kreuzung zählt so viele Kerzen lang
    sma_kreuz_fenster: int = 10           # Golden/Death Cross zählt so viele Kerzen lang
    min_bestaetigungen: int = 4           # Mindestanzahl erfüllter Zusatzbedingungen für ein Signal
    schwelle_stark_positiv: int = 6
    schwelle_positiv: int = 3
    schwelle_negativ: int = -3
    schwelle_stark_negativ: int = -6
    gewichte: PunkteGewichte = field(default_factory=PunkteGewichte)

    def pruefen(self) -> list[str]:
        """Prüft die Parameter auf Widersprüche und liefert verständliche Fehlermeldungen."""
        fehler: list[str] = []
        if not (self.schwelle_stark_negativ < self.schwelle_negativ < self.schwelle_positiv
                < self.schwelle_stark_positiv):
            fehler.append("Die Bewertungsschwellen müssen aufsteigend sein: "
                          "stark negativ < negativ < positiv < stark positiv.")
        if not (self.schwelle_negativ < 0 < self.schwelle_positiv):
            fehler.append("Die Schwelle „negativ“ muss unter 0 und die Schwelle „positiv“ über 0 liegen.")
        if not (0 < self.rsi_ueberverkauft < self.rsi_ueberkauft < 100):
            fehler.append("Die RSI-Schwellen müssen gelten: 0 < überverkauft < überkauft < 100.")
        if not (self.rsi_ueberverkauft < self.rsi_kauf_obergrenze <= 100):
            fehler.append("Die RSI-Obergrenze für Kaufsignale muss über der Überverkauft-Schwelle liegen.")
        if self.macd_fenster < 1 or self.sma_kreuz_fenster < 1:
            fehler.append("Die Signalfenster müssen mindestens 1 Kerze betragen.")
        return fehler


@dataclass
class RisikoParameter:
    """Parameter für Risikoanalyse und Backtest."""

    kapital: float = 100_000.0
    risiko_pro_position_pct: float = 1.0
    atr_multiplikator: float = 2.0
    crv: float = 2.0
    transaktionskosten_pct: float = 0.10  # je Kauf bzw. Verkauf, in % des Handelswerts
    slippage_pct: float = 0.05            # je Ausführung, in % des Kurses
    risikofreier_zins_pct: float = 0.0    # p. a., für die Sharpe Ratio


@dataclass
class BacktestParameter:
    """Einstellungen des Backtests."""

    stop_loss_aktiv: bool = True
    kursziel_aktiv: bool = True
    positionsgroesse: str = "voll"        # "voll" = gesamtes Kapital, "risiko" = risikobasiert
    nur_starke_signale: bool = False


# =============================================================================
# 2) Hilfsfunktionen: Formatierung (deutsches Zahlenformat)
# =============================================================================

def _ist_zahl(wert: Any) -> bool:
    try:
        return wert is not None and math.isfinite(float(wert))
    except (TypeError, ValueError):
        return False


def fmt_zahl(wert: Any, nachkomma: int = 2, vorzeichen: bool = False) -> str:
    """Formatiert Zahlen im deutschen Format: 1.234,56."""
    if not _ist_zahl(wert):
        return "–"
    muster = f"{{:{'+' if vorzeichen else ''},.{nachkomma}f}}"
    text = muster.format(float(wert))
    return text.replace(",", "§").replace(".", ",").replace("§", ".")


def fmt_pct(anteil: Any, nachkomma: int = 2, vorzeichen: bool = True) -> str:
    """Formatiert einen Anteil (0,0123) als Prozentwert (+1,23 %)."""
    if not _ist_zahl(anteil):
        return "–"
    return f"{fmt_zahl(float(anteil) * 100, nachkomma, vorzeichen)} %"


def fmt_volumen(wert: Any) -> str:
    """Kompakte Darstellung großer Stückzahlen/Beträge (Tsd., Mio., Mrd.)."""
    if not _ist_zahl(wert):
        return "–"
    wert = float(wert)
    for grenze, einheit in ((1e9, "Mrd."), (1e6, "Mio."), (1e3, "Tsd.")):
        if abs(wert) >= grenze:
            return f"{fmt_zahl(wert / grenze, 2)} {einheit}"
    return fmt_zahl(wert, 0)


def fmt_spanne(unten: Any, oben: Any) -> str:
    """Kompakte Spanne „27,73–57,96“ (ab 1.000 ohne Nachkommastellen, z. B. bei Indizes)."""
    stellen = 0 if _ist_zahl(oben) and float(oben) >= 1000 else 2
    return f"{fmt_zahl(unten, stellen)}–{fmt_zahl(oben, stellen)}"


def fmt_datum(zeitpunkt: Any, intraday: bool = False) -> str:
    if zeitpunkt is None or (isinstance(zeitpunkt, float) and math.isnan(zeitpunkt)):
        return "–"
    ts = pd.Timestamp(zeitpunkt)
    return ts.strftime("%d.%m.%Y %H:%M" if intraday else "%d.%m.%Y")


def fmt_punkte(punkte: Any) -> str:
    """Punktzahl mit Vorzeichen („+3“, „−2“, „0“)."""
    n = int(punkte)
    return f"{n:+d}" if n else "0"


def punkte_text(punkte: Any) -> str:
    """„+1 Punkt“, „+4 Punkte“, „0 Punkte“."""
    n = int(punkte)
    return f"{fmt_punkte(n)} Punkt" if abs(n) == 1 else f"{fmt_punkte(n)} Punkte"


def _jetzt_istanbul() -> pd.Timestamp:
    """Aktuelle Uhrzeit in Istanbul (ohne Zeitzonenangabe, passend zu den Kursdaten)."""
    return pd.Timestamp.now(tz=ZEITZONE_BIST).tz_localize(None)


# =============================================================================
# 3) Daten: Ticker, Laden, Prüfen
# =============================================================================

class DatenFehler(Exception):
    """Verständliche Fehlermeldung beim Laden oder Prüfen von Kursdaten."""

    def __init__(self, meldung: str, art: str = "allgemein", tipps: Sequence[str] | None = None):
        super().__init__(meldung)
        self.meldung = meldung
        self.art = art            # eingabe | keine_daten | netzwerk | limit | abhaengigkeit | allgemein
        self.tipps = list(tipps or [])


@dataclass(frozen=True)
class TickerInfo:
    """Ergebnis der Ticker-Normalisierung für die verschiedenen Datenquellen."""

    eingabe: str
    basis: str        # Börsenkürzel ohne Suffix, z. B. THYAO oder XU100
    ist_index: bool
    yahoo: str        # Symbol für Yahoo Finance, z. B. THYAO.IS
    borsapy: str      # Symbol für borsapy/TradingView, z. B. THYAO


_TUERKISCH_ASCII = str.maketrans({
    "ç": "c", "Ç": "C", "ğ": "g", "Ğ": "G", "ı": "i", "İ": "I", "ö": "o", "Ö": "O",
    "ş": "s", "Ş": "S", "ü": "u", "Ü": "U", "â": "a", "Â": "A", "î": "i", "Î": "I", "û": "u", "Û": "U",
})


def _ascii_falten(text: str) -> str:
    """Türkische Sonderzeichen in ASCII umwandeln und in Kleinbuchstaben setzen (für Vergleiche)."""
    return unicodedata.normalize("NFC", str(text)).translate(_TUERKISCH_ASCII).lower().strip()


def ticker_normalisieren(eingabe: str, suffix: str = STANDARD_SUFFIX, unveraendert: bool = False) -> TickerInfo:
    """Wandelt Benutzereingaben in die Symbolformate der Datenanbieter um.

    Beispiele: ``thyao`` → THYAO.IS, ``BIST:THYAO`` → THYAO.IS, ``THYAO.E`` → THYAO.IS,
    ``ŞİŞE`` → SISE.IS, ``BIST 100`` → XU100.IS. Anbieter formatieren BIST-Symbole unterschiedlich:
    Yahoo Finance ``THYAO.IS``, TradingView ``BIST:THYAO``, Bloomberg ``THYAO TI``, Foreks/Matriks ``THYAO.E``.
    """
    roh = unicodedata.normalize("NFC", str(eingabe or "")).strip()
    if not roh:
        raise DatenFehler("Bitte einen Ticker eingeben (z. B. THYAO).", art="eingabe")

    if unveraendert:  # Symbol exakt so verwenden, wie es eingegeben wurde (z. B. für andere Börsen)
        symbol = roh.upper()
        basis = re.sub(r"\.[A-Z]{1,4}$", "", symbol)
        return TickerInfo(roh, basis, basis in BIST_INDIZES, symbol, basis)

    s = re.sub(r"\s+", " ", roh.translate(_TUERKISCH_ASCII).upper()).strip()
    kompakt = s.replace(" ", "").replace("-", "")
    if kompakt in INDEX_ALIASE:
        s = INDEX_ALIASE[kompakt]
    s = re.sub(r"^(BIST|XIST|IST|BORSAISTANBUL)\s*:\s*", "", s)   # TradingView / Google Finance
    s = re.sub(r"\s+TI(\s+EQUITY)?$", "", s)                      # Bloomberg
    s = re.sub(r"\.(IS|E|IST|TI)$", "", s)                         # Yahoo/Reuters, Foreks/Matriks
    s = s.replace(" ", "")
    s = INDEX_ALIASE.get(s, s)
    if not re.fullmatch(r"[A-Z0-9]{2,10}", s):
        raise DatenFehler(
            f"„{roh}“ ist kein gültiges Börsenkürzel.",
            art="eingabe",
            tipps=["Bitte das Kürzel der Borsa İstanbul verwenden, z. B. THYAO, GARAN oder XU100.",
                   *[f"Meinten Sie {v}?" for v in ticker_vorschlaege(roh)]],
        )
    suffix = (suffix or "").strip().upper()
    if suffix and not suffix.startswith("."):
        suffix = "." + suffix
    ist_index = s in BIST_INDIZES or bool(re.fullmatch(r"X[A-Z][A-Z0-9]{3}", s))
    return TickerInfo(roh, s, ist_index, s + suffix, s)


def ticker_vorschlaege(eingabe: str, anzahl: int = 3) -> list[str]:
    """Schlägt anhand von Kürzeln und Firmennamen passende Ticker vor (offline)."""
    suchbegriff = _ascii_falten(eingabe)
    if not suchbegriff:
        return []
    katalog = {**BIST_FAVORITEN, **BIST_INDIZES}
    treffer: list[str] = []
    for kuerzel, name in katalog.items():
        if suchbegriff in _ascii_falten(name) or suchbegriff in kuerzel.lower():
            treffer.append(f"{kuerzel} ({name})")
    if len(treffer) < anzahl:
        namen = {_ascii_falten(n): k for k, n in katalog.items()}
        namen.update({k.lower(): k for k in katalog})
        for passend in difflib.get_close_matches(suchbegriff, list(namen), n=anzahl, cutoff=0.6):
            kuerzel = namen[passend]
            eintrag = f"{kuerzel} ({katalog[kuerzel]})"
            if eintrag not in treffer:
                treffer.append(eintrag)
    return treffer[:anzahl]


def _keine_daten_fehler(symbol: str, quelle: str, eingabe: str = "") -> DatenFehler:
    tipps = [
        "Börsenkürzel prüfen: BIST-Aktien werden bei Yahoo Finance mit dem Suffix „.IS“ geführt "
        "(z. B. THYAO.IS). Die App ergänzt das Suffix automatisch.",
        "Firmennamen funktionieren nicht – bitte das Kürzel verwenden (z. B. SISE statt Şişecam).",
        "Nicht jede BIST-Aktie und nicht jeder Index ist bei jedem Anbieter verfügbar. Alternativen: "
        "Datenquelle „borsapy / TradingView“ oder eine CSV-Datei Ihres Brokers.",
        "Bei neu gelisteten, umbenannten oder vom Handel ausgesetzten Werten kann die Historie fehlen.",
    ]
    tipps += [f"Meinten Sie {v}?" for v in ticker_vorschlaege(eingabe or symbol.split(".")[0])]
    return DatenFehler(f"Für „{symbol}“ wurden bei {quelle} keine Kursdaten gefunden.",
                       art="keine_daten", tipps=tipps)


def _fehler_uebersetzen(exc: BaseException, symbol: str, quelle: str, eingabe: str = "") -> DatenFehler:
    """Übersetzt technische Ausnahmen der Datenbibliotheken in verständliche Meldungen."""
    if isinstance(exc, DatenFehler):
        return exc
    name = type(exc).__name__
    text = str(exc)
    t = f"{name} {text}".lower()
    if "ratelimit" in t or "too many requests" in t or " 429" in t:
        return DatenFehler(f"{quelle} begrenzt derzeit die Anzahl der Anfragen (Rate-Limit).", art="limit",
                           tipps=["Einige Minuten warten und erneut versuchen.",
                                  "Zwischenzeitlich eine andere Datenquelle verwenden."])
    netzwerk = ("connection", "timeout", "timed out", "resolve", "network", "unreachable", "curl",
                "failed to perform", "ssl", "proxy", "max retries", "getaddrinfo", "name or service",
                "websocket", "tunnel")
    if any(stichwort in t for stichwort in netzwerk):
        return DatenFehler(f"Keine Verbindung zu {quelle} möglich.", art="netzwerk",
                           tipps=["Internetverbindung, Firewall oder Proxy prüfen.",
                                  "Später erneut versuchen – der Dienst kann vorübergehend gestört sein.",
                                  "Offline testen: Datenquelle „Demo-Daten“ oder eigene CSV-Datei.",
                                  f"Technische Details: {name}: {text[:200]}"])
    keine_daten = ("delisted", "no price data", "no data", "not found", "no timezone", "symbol",
                   "tickermissing", "pricesmissing", "empty")
    if any(stichwort in t for stichwort in keine_daten):
        return _keine_daten_fehler(symbol, quelle, eingabe)
    return DatenFehler(f"Beim Laden der Daten von {quelle} ist ein Fehler aufgetreten.",
                       tipps=[f"Technische Details: {name}: {text[:300]}"])


@dataclass
class DatenPaket:
    """Ergebnis von daten_laden(): Rohdaten inklusive Vorlauf und Metadaten."""

    daten: pd.DataFrame              # OHLCV in Lade-Frequenz (täglich bzw. intraday) inkl. Vorlauf
    ticker: TickerInfo
    symbol: str                      # tatsächlich abgefragtes Symbol
    quelle: str
    intervall: str
    zeitraum: str
    anzeige_start: pd.Timestamp      # Beginn des gewählten Anzeigezeitraums
    name: str
    waehrung: str
    ist_index: bool
    meta: dict[str, Any] = field(default_factory=dict)
    hinweise: list[str] = field(default_factory=list)


OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def _index_bereinigen(df: pd.DataFrame, intraday: bool) -> pd.DataFrame:
    """Zeitzone entfernen (lokale Börsenzeit bleibt erhalten), Tagesdaten auf Datum normieren, sortieren."""
    daten = df.copy()
    idx = pd.DatetimeIndex(pd.to_datetime(daten.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    if not intraday:
        idx = idx.normalize()
    daten.index = idx
    daten.index.name = "Datum"
    return daten[~daten.index.duplicated(keep="last")].sort_index()


def _yahoo_laden(symbol: str, start: pd.Timestamp, ende: pd.Timestamp, intervall_code: str,
                 dividendenbereinigt: bool = True, eingabe: str = "") -> tuple[pd.DataFrame, dict[str, Any]]:
    """Lädt Kursdaten über yfinance (Yahoo Finance). ``ende`` ist exklusiv."""
    try:
        import logging

        import yfinance as yf
    except ImportError as exc:
        raise DatenFehler("Das Paket „yfinance“ ist nicht installiert.", art="abhaengigkeit",
                          tipps=["Installation: pip install -r requirements.txt"]) from exc
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)   # Konsolenausgaben unterdrücken
    with suppress(Exception):
        yf.config.debug.hide_exceptions = False                 # yfinance ≥ 1.0: Fehler als Ausnahme melden
    intraday = intervall_code not in ("1d", "1wk", "1mo")
    ticker = yf.Ticker(symbol)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = ticker.history(start=start.strftime("%Y-%m-%d"), end=ende.strftime("%Y-%m-%d"),
                                interval=intervall_code, auto_adjust=dividendenbereinigt, actions=True)
    except Exception as exc:
        raise _fehler_uebersetzen(exc, symbol, "Yahoo Finance", eingabe) from exc
    if df is None or df.empty or "Close" not in df.columns or df["Close"].dropna().empty:
        raise _keine_daten_fehler(symbol, "Yahoo Finance", eingabe)

    meta: dict[str, Any] = {}
    try:  # Metadaten sind optional (Name, Währung, Börse, Instrumenttyp)
        roh_meta = ticker.history_metadata or {}
        for schluessel in ("longName", "shortName", "currency", "exchangeName", "fullExchangeName",
                           "instrumentType", "exchangeTimezoneName", "fiftyTwoWeekHigh", "fiftyTwoWeekLow"):
            wert = roh_meta.get(schluessel)
            if wert not in (None, ""):
                meta[schluessel] = wert
    except Exception:
        pass
    spalten = [s for s in (*OHLCV, "Dividends", "Stock Splits") if s in df.columns]
    return _index_bereinigen(df[spalten], intraday), meta


def yahoo_mehrere_laden(basis_liste: Sequence[str], start: pd.Timestamp, ende: pd.Timestamp,
                        dividendenbereinigt: bool = True, paketgroesse: int = 80) -> dict[str, pd.DataFrame]:
    """Lädt Tageskurse vieler BIST-Aktien gebündelt über yf.download (deutlich schneller als Einzelabrufe).

    Rückgabe: {Basiskürzel: OHLCV-DataFrame}; Werte ohne Daten fehlen im Ergebnis.
    """
    try:
        import logging

        import yfinance as yf
    except ImportError as exc:
        raise DatenFehler("Das Paket „yfinance“ ist nicht installiert.", art="abhaengigkeit") from exc
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    ergebnis: dict[str, pd.DataFrame] = {}
    liste = list(dict.fromkeys(basis_liste))
    fehler: Exception | None = None
    for i in range(0, len(liste), paketgroesse):
        teil = liste[i:i + paketgroesse]
        symbole = [f"{b}{STANDARD_SUFFIX}" for b in teil]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = yf.download(symbole, start=start.strftime("%Y-%m-%d"), end=ende.strftime("%Y-%m-%d"),
                                 interval="1d", auto_adjust=dividendenbereinigt, group_by="ticker",
                                 threads=True, progress=False)
        except Exception as exc:   # ein Paket darf nicht alles abbrechen
            fehler = exc
            continue
        if df is None or df.empty:
            continue
        for basis, symbol in zip(teil, symbole, strict=True):
            if isinstance(df.columns, pd.MultiIndex):
                if symbol not in df.columns.get_level_values(0):
                    continue
                einzeln = df[symbol]
            else:          # nur ein Symbol angefragt
                einzeln = df
            einzeln = einzeln[[s for s in OHLCV if s in einzeln.columns]].dropna(how="all")
            if not einzeln.empty and einzeln["Close"].notna().sum() > 0:
                ergebnis[basis] = _index_bereinigen(einzeln, intraday=False)
    if not ergebnis and fehler is not None:
        raise _fehler_uebersetzen(fehler, "BIST-Aktien", "Yahoo Finance")
    return ergebnis


def _borsapy_laden(symbol: str, start: pd.Timestamp, ende: pd.Timestamp, intervall_code: str,
                   eingabe: str = "") -> tuple[pd.DataFrame, dict[str, Any]]:
    """Lädt Kursdaten über das optionale Paket borsapy (TradingView-Daten, ca. 15 Min. verzögert)."""
    try:
        import borsapy as bp  # type: ignore[import-not-found]
    except ImportError as exc:
        raise DatenFehler("Das optionale Paket „borsapy“ ist nicht installiert.", art="abhaengigkeit",
                          tipps=["Installation: pip install borsapy",
                                 "Alternativ die Datenquelle Yahoo Finance verwenden."]) from exc
    intraday = intervall_code not in ("1d", "1wk")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = bp.Ticker(symbol).history(start=start.strftime("%Y-%m-%d"),
                                           end=(ende - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                                           interval=intervall_code)
    except Exception as exc:
        raise _fehler_uebersetzen(exc, symbol, "borsapy/TradingView", eingabe) from exc
    if df is None or df.empty or "Close" not in df.columns:
        raise _keine_daten_fehler(symbol, "borsapy/TradingView", eingabe)
    spalten = [s for s in OHLCV if s in df.columns]
    return _index_bereinigen(df[spalten], intraday), {"currency": "TRY", "exchangeName": "BIST"}


# --- CSV-Import (deutsche, türkische und englische Exporte, z. B. Broker, Investing.com, yfinance) ---
_CSV_SPALTEN: dict[str, tuple[str, ...]] = {
    "Datum": ("date", "datum", "tarih", "datetime", "timestamp", "zeitstempel", "time", "zeit", "gun", "tarih/saat"),
    "Open": ("open", "eroffnung", "eroeffnung", "eroffn.", "eroffn", "eroffnungskurs", "acilis",
             "acilis fiyati", "opening", "erster"),
    "High": ("high", "hoch", "hochst", "hoechst", "tageshoch", "yuksek", "en yuksek", "max"),
    "Low": ("low", "tief", "tiefst", "tagestief", "dusuk", "en dusuk", "min"),
    "Close": ("close", "schluss", "schlusskurs", "kapanis", "kapanis fiyati", "zuletzt", "last", "son",
              "simdi", "price", "kurs", "fiyat", "adj close", "adj. close"),
    "Volume": ("volume", "volumen", "vol.", "vol", "hacim", "hac.", "hacim (lot)", "lot", "stuck", "stueck"),
}


def _spalte_finden(spalten: Sequence[str], aliase: Sequence[str]) -> str | None:
    gefaltet = {_ascii_falten(s).strip(" \"'"): s for s in spalten}
    for alias in aliase:
        treffer = gefaltet.get(_ascii_falten(alias))
        if treffer is not None:
            return treffer
    return None


def _dezimalzeichen_erkennen(werte: pd.Series) -> str:
    """Erkennt das Dezimaltrennzeichen einer Kursspalte (',' für 1.234,56 – '.' für 1,234.56)."""
    s = werte.dropna().astype(str).str.strip().str.replace(r"[^\d,.\-]", "", regex=True)
    beide = s[s.str.contains(",", regex=False) & s.str.contains(".", regex=False)]
    if not beide.empty:
        return "," if (beide.str.rfind(",") > beide.str.rfind(".")).mean() >= 0.5 else "."
    mit_komma = s[s.str.contains(",", regex=False)]
    if not mit_komma.empty:
        nur_tausender = mit_komma.str.fullmatch(r"-?\d{1,3}(,\d{3})+").all()
        return "." if nur_tausender else ","
    if (s.str.count(r"\.") > 1).any():   # 1.234.567 → Punkte sind Tausendertrennzeichen
        return ","
    return "."


def _zahlen_parsen(werte: pd.Series, dezimal: str = ".") -> pd.Series:
    """Wandelt Zahlen aus Textdateien um (inkl. Einheiten K/M/B wie bei Investing.com: 12,5M)."""
    if pd.api.types.is_numeric_dtype(werte):
        return pd.to_numeric(werte, errors="coerce").astype(float)
    s = (werte.astype(str).str.strip().str.replace(" ", "", regex=False)
         .str.replace(" ", "", regex=False).str.strip("\"'").str.replace("%", "", regex=False))
    einheit = s.str.extract(r"([KkMmBb])$", expand=False).str.upper()
    faktor = einheit.map({"K": 1e3, "M": 1e6, "B": 1e9}).astype(float).fillna(1.0)
    s = s.str.replace(r"[KkMmBb]$", "", regex=True)
    if dezimal == ",":
        s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    else:
        s = s.str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce").astype(float) * faktor


def _datum_parsen(werte: pd.Series) -> pd.DatetimeIndex:
    """Erkennt ISO-, deutsche/türkische (TT.MM.JJJJ) und US-Datumsformate (MM/TT/JJJJ)."""
    if pd.api.types.is_datetime64_any_dtype(werte):
        return pd.DatetimeIndex(werte)
    s = werte.astype(str).str.strip().str.strip("\"'")
    if s.str.contains(r"(?:[+-]\d{2}:?\d{2}|Z)$", regex=True).mean() > 0.5:   # mit Zeitzone (z. B. yfinance)
        ts = pd.to_datetime(s, errors="coerce", utc=True, format="ISO8601")
        return pd.DatetimeIndex(ts).tz_convert(ZEITZONE_BIST).tz_localize(None)
    if s.str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.8:
        return pd.DatetimeIndex(pd.to_datetime(s, errors="coerce", format="ISO8601"))
    tag_zuerst = True
    teile = s.str.extract(r"^(\d{1,2})[./-](\d{1,2})[./-]\d{2,4}")
    if teile.notna().all(axis=1).mean() > 0.8:
        erster = pd.to_numeric(teile[0], errors="coerce")
        zweiter = pd.to_numeric(teile[1], errors="coerce")
        if (erster > 12).any():
            tag_zuerst = True
        elif (zweiter > 12).any():
            tag_zuerst = False          # US-Format MM/TT/JJJJ
    return pd.DatetimeIndex(pd.to_datetime(s, errors="coerce", dayfirst=tag_zuerst, format="mixed"))


def _csv_einlesen(inhalt: bytes) -> tuple[pd.DataFrame, list[str]]:
    """Liest eine CSV-Datei mit Kursdaten. Trennzeichen, Zahlen- und Datumsformat werden erkannt."""
    if not inhalt:
        raise DatenFehler("Die CSV-Datei ist leer.", art="eingabe")
    text = ""
    for kodierung in ("utf-8-sig", "cp1254", "cp1252", "latin-1"):
        try:
            text = inhalt.decode(kodierung)
            break
        except UnicodeDecodeError:
            continue
    zeilen = [z for z in text.splitlines() if z.strip()]
    if len(zeilen) < 2:
        raise DatenFehler("Die CSV-Datei enthält keine Datenzeilen.", art="eingabe")
    kopf = zeilen[0]
    trenner = max([";", ",", "\t", "|"], key=kopf.count)
    if kopf.count(trenner) == 0:
        raise DatenFehler("Das Trennzeichen der CSV-Datei wurde nicht erkannt.", art="eingabe",
                          tipps=["Erlaubt sind Semikolon, Komma, Tabulator oder senkrechter Strich."])
    try:
        roh = pd.read_csv(io.StringIO(text), sep=trenner, dtype=str, skipinitialspace=True, engine="python")
    except Exception as exc:
        raise DatenFehler("Die CSV-Datei konnte nicht gelesen werden.", art="eingabe",
                          tipps=[f"Technische Details: {type(exc).__name__}: {exc}"]) from exc
    roh.columns = [str(c).strip().strip("\"'") for c in roh.columns]
    zuordnung = {ziel: _spalte_finden(list(roh.columns), aliase) for ziel, aliase in _CSV_SPALTEN.items()}
    if zuordnung["Datum"] is None and (roh.columns[0] == "" or roh.columns[0].startswith("Unnamed")):
        zuordnung["Datum"] = roh.columns[0]
    erwartet = ("Erwartete Spalten (Beispiele): Datum/Date/Tarih, Eröffnung/Open/Açılış, Hoch/High/Yüksek, "
                "Tief/Low/Düşük, Schluss/Close/Kapanış, Volumen/Volume/Hacim.")
    if zuordnung["Datum"] is None or zuordnung["Close"] is None:
        raise DatenFehler("In der CSV-Datei wurden keine Datums- bzw. Schlusskurs-Spalte gefunden.",
                          art="eingabe", tipps=[erwartet, f"Gefundene Spalten: {', '.join(roh.columns)}"])
    dezimal = _dezimalzeichen_erkennen(roh[zuordnung["Close"]])
    daten = pd.DataFrame(index=roh.index)
    for ziel in OHLCV:
        spalte = zuordnung[ziel]
        daten[ziel] = _zahlen_parsen(roh[spalte], dezimal) if spalte else np.nan
    daten.index = _datum_parsen(roh[zuordnung["Datum"]])
    daten = daten[daten.index.notna()]
    if daten.empty or daten["Close"].notna().sum() == 0:
        raise DatenFehler("Die CSV-Datei enthält keine auswertbaren Kurse.", art="eingabe", tipps=[erwartet])

    hinweise: list[str] = []
    fehlend = [z for z in ("Open", "High", "Low") if zuordnung[z] is None]
    if fehlend:
        hinweise.append(f"CSV: Spalten fehlen ({', '.join(fehlend)}) – ersatzweise wird der Schlusskurs verwendet.")
    if zuordnung["Volume"] is None:
        hinweise.append("CSV: Keine Volumenspalte gefunden – volumenbasierte Auswertungen sind deaktiviert.")
    intraday = bool((daten.index != daten.index.normalize()).any())
    daten = _index_bereinigen(daten, intraday=True)
    if intraday:   # Intraday-Export → zu Tageskerzen verdichten
        daten = (daten.groupby(daten.index.normalize())
                 .agg(Open=("Open", "first"), High=("High", "max"), Low=("Low", "min"),
                      Close=("Close", "last"), Volume=("Volume", "sum")))
        daten.index.name = "Datum"
        hinweise.append("CSV: Intraday-Daten wurden zu Tageskerzen verdichtet.")
    return daten, hinweise


def _demo_daten_erzeugen(basis: str, start: pd.Timestamp, ende: pd.Timestamp, intervall: IntervallInfo,
                         jetzt: pd.Timestamp | None = None) -> pd.DataFrame:
    """Erzeugt reproduzierbare, synthetische Kursdaten (KEINE echten Kurse) für Tests ohne Internet."""
    rng = np.random.default_rng(zlib.crc32(basis.encode("utf-8")))
    tage = pd.bdate_range(start.normalize(), (ende - pd.Timedelta(days=1)).normalize())
    if len(tage) < 5:
        tage = pd.bdate_range(end=ende.normalize(), periods=60)
    n = len(tage)
    drift, vola = np.empty(n), np.empty(n)
    regime = [(0.0016, 0.017), (-0.0013, 0.025), (0.0002, 0.012), (0.0009, 0.021)]  # Trend-/Seitwärtsphasen
    i = 0
    while i < n:
        laenge = int(rng.integers(35, 110))
        mu, sigma = regime[int(rng.integers(0, len(regime)))]
        drift[i:i + laenge], vola[i:i + laenge] = mu, sigma
        i += laenge
    startkurs = float(rng.uniform(40, 320))

    if not intervall.intraday:
        renditen = np.clip(drift + vola * rng.standard_t(5, n) / math.sqrt(5 / 3), -0.095, 0.095)
        schluss = startkurs * np.exp(np.cumsum(renditen))
        eroeffnung = np.r_[startkurs, schluss[:-1]] * np.exp(rng.normal(0, 0.3, n) * vola)
        hoch = np.maximum(eroeffnung, schluss) * np.exp(np.abs(rng.normal(0, 0.55, n)) * vola)
        tief = np.minimum(eroeffnung, schluss) * np.exp(-np.abs(rng.normal(0, 0.55, n)) * vola)
        volumen = rng.lognormal(np.log(8e6), 0.35, n) * (1 + 18 * np.abs(renditen))
        index = tage
    else:
        pro_tag = int(480 / (intervall.minuten or 60))          # Handelszeit 10:00–18:00 Uhr
        index = pd.DatetimeIndex([tag + pd.Timedelta(minutes=600 + k * (intervall.minuten or 60))
                                  for tag in tage for k in range(pro_tag)])
        m = len(index)
        vola_kerze = np.repeat(vola, pro_tag) / math.sqrt(pro_tag)
        renditen = np.repeat(drift, pro_tag) / pro_tag + vola_kerze * rng.standard_normal(m)
        schluss = startkurs * np.exp(np.cumsum(renditen))
        eroeffnung = np.r_[startkurs, schluss[:-1]]
        hoch = np.maximum(eroeffnung, schluss) * np.exp(np.abs(rng.normal(0, 0.5, m)) * vola_kerze)
        tief = np.minimum(eroeffnung, schluss) * np.exp(-np.abs(rng.normal(0, 0.5, m)) * vola_kerze)
        volumen = rng.lognormal(np.log(8e6 / pro_tag), 0.5, m) * (1 + 10 * np.abs(renditen) / vola_kerze.mean())

    df = pd.DataFrame({"Open": eroeffnung, "High": hoch, "Low": tief, "Close": schluss,
                       "Volume": np.round(volumen)}, index=index).round({"Open": 2, "High": 2, "Low": 2, "Close": 2})
    df["High"] = df[["High", "Open", "Close"]].max(axis=1)
    df["Low"] = df[["Low", "Open", "Close"]].min(axis=1)
    df.index.name = "Datum"
    if jetzt is not None:
        df = df[df.index <= jetzt]
    return df


def daten_laden(
    eingabe: str,
    zeitraum: str = "1 Jahr",
    intervall: str = "Täglich",
    quelle: str = QUELLE_YAHOO,
    suffix: str = STANDARD_SUFFIX,
    symbol_unveraendert: bool = False,
    dividendenbereinigt: bool = True,
    csv_inhalt: bytes | None = None,
    jetzt: pd.Timestamp | None = None,
) -> DatenPaket:
    """Lädt Kursdaten für den gewählten Zeitraum plus Vorlauf für die Indikatorberechnung.

    * Für „Wöchentlich“ werden Tagesdaten geladen und später zu Wochenkerzen verdichtet
      (Datum = letzter Handelstag der Woche).
    * Intraday-Daten sind bei Yahoo Finance zeitlich begrenzt (60 Tage bzw. 730 Tage bei 60 Minuten);
      der Zeitraum wird dann automatisch gekürzt und ein Hinweis erzeugt.
    """
    if zeitraum not in ZEITRAEUME:
        raise DatenFehler(f"Unbekannter Zeitraum „{zeitraum}“.", art="eingabe")
    if intervall not in INTERVALLE:
        raise DatenFehler(f"Unbekanntes Intervall „{intervall}“.", art="eingabe")
    info = INTERVALLE[intervall]
    jetzt = jetzt if jetzt is not None else _jetzt_istanbul()
    hinweise: list[str] = []

    if quelle in (QUELLE_CSV, QUELLE_DEMO):
        try:
            tinfo = ticker_normalisieren(eingabe or "CSV", suffix, symbol_unveraendert)
        except DatenFehler:
            basis = re.sub(r"[^A-Z0-9]", "", _ascii_falten(eingabe).upper())[:10] or "CSV"
            tinfo = TickerInfo(eingabe, basis, False, basis, basis)
    else:
        tinfo = ticker_normalisieren(eingabe, suffix, symbol_unveraendert)

    anzeige_tage = ZEITRAEUME[zeitraum]
    if info.intraday and quelle == QUELLE_CSV:
        raise DatenFehler("Für CSV-Dateien stehen die Intervalle „Täglich“ und „Wöchentlich“ zur Verfügung.",
                          art="eingabe")
    if info.max_tage is not None and anzeige_tage + info.vorlauf_tage > info.max_tage:
        neu = max(info.max_tage - info.vorlauf_tage, 1)
        hinweise.append(f"Für „{intervall}“ stellen die Datenquellen nur etwa {info.max_tage} Tage Historie "
                        f"bereit. Der Anzeigezeitraum wurde auf {neu} Tage gekürzt (plus Vorlauf für Indikatoren).")
        anzeige_tage = neu
    lade_code = info.code if info.intraday else "1d"
    start = (jetzt - pd.Timedelta(days=anzeige_tage + info.vorlauf_tage)).normalize()
    ende = jetzt.normalize() + pd.Timedelta(days=1)          # exklusiv → heute ist enthalten
    anzeige_start = (jetzt - pd.Timedelta(days=anzeige_tage)).normalize()
    meta: dict[str, Any] = {}

    if quelle == QUELLE_YAHOO:
        daten, meta = _yahoo_laden(tinfo.yahoo, start, ende, lade_code, dividendenbereinigt, eingabe)
        symbol = tinfo.yahoo
    elif quelle == QUELLE_BORSAPY:
        daten, meta = _borsapy_laden(tinfo.borsapy, start, ende, lade_code, eingabe)
        symbol = tinfo.borsapy
    elif quelle == QUELLE_CSV:
        if not csv_inhalt:
            raise DatenFehler("Bitte in der Seitenleiste eine CSV-Datei hochladen.", art="eingabe",
                              tipps=["Die Datei braucht mindestens eine Datums- und eine Schlusskurs-Spalte."])
        daten, csv_hinweise = _csv_einlesen(csv_inhalt)
        hinweise += csv_hinweise
        anzeige_start = (daten.index.max() - pd.Timedelta(days=anzeige_tage)).normalize()
        daten = daten.loc[daten.index >= anzeige_start - pd.Timedelta(days=info.vorlauf_tage)]
        symbol = tinfo.basis
        meta = {"currency": "TRY"}
        hinweise.append("CSV-Daten: Als Währung wird TRY angenommen; der Zeitraum bezieht sich auf das "
                        "letzte Datum der Datei.")
    elif quelle == QUELLE_DEMO:
        daten = _demo_daten_erzeugen(tinfo.basis, start, ende, info, jetzt)
        symbol = tinfo.basis
        meta = {"currency": "TRY"}
        hinweise.append("DEMO-DATEN: synthetisch erzeugte Kurse – keine echten Marktdaten!")
    else:
        raise DatenFehler(f"Unbekannte Datenquelle „{quelle}“.", art="eingabe")

    if daten.empty or daten["Close"].dropna().empty:
        raise _keine_daten_fehler(symbol, quelle, eingabe)
    if (daten.index >= anzeige_start).sum() < 2:
        raise DatenFehler(
            f"Im gewählten Zeitraum liegen für „{symbol}“ keine Kursdaten vor "
            f"(letzte verfügbare Kerze: {fmt_datum(daten.index.max(), info.intraday)}).",
            art="keine_daten",
            tipps=["Einen längeren Zeitraum wählen.",
                   "Der Wert könnte vom Handel ausgesetzt oder delistet sein."])

    ist_index = tinfo.ist_index or str(meta.get("instrumentType", "")).upper() == "INDEX"
    name = str(meta.get("longName") or meta.get("shortName") or BIST_FAVORITEN.get(tinfo.basis)
               or BIST_INDIZES.get(tinfo.basis) or BIST_ALLE_AKTIEN.get(tinfo.basis) or tinfo.basis)
    if quelle == QUELLE_DEMO:
        name = f"DEMO – {name}"
    return DatenPaket(daten=daten, ticker=tinfo, symbol=symbol, quelle=quelle, intervall=intervall,
                      zeitraum=zeitraum, anzeige_start=anzeige_start, name=name,
                      waehrung=str(meta.get("currency") or "TRY").upper(), ist_index=ist_index,
                      meta=meta, hinweise=hinweise)


@dataclass
class DatenQualitaet:
    """Prüfbericht von daten_validieren()."""

    zeilen_roh: int = 0
    zeilen_bereinigt: int = 0
    entfernte_zeilen: int = 0
    fehlende_werte: dict[str, int] = field(default_factory=dict)
    korrigierte_ohlc: int = 0
    luecken: list[dict[str, Any]] = field(default_factory=list)
    kurs_spruenge: pd.DataFrame = field(default_factory=pd.DataFrame)
    handelsaussetzungen: int = 0
    volumen_verfuegbar: bool = True
    null_volumen_anteil: float = 0.0
    erste_kerze: pd.Timestamp | None = None
    letzte_kerze: pd.Timestamp | None = None
    warnungen: list[str] = field(default_factory=list)
    hinweise: list[str] = field(default_factory=list)


def daten_validieren(df: pd.DataFrame, intervall: IntervallInfo, jetzt: pd.Timestamp | None = None,
                     live: bool = True) -> tuple[pd.DataFrame, DatenQualitaet]:
    """Prüft und bereinigt Kursdaten.

    * entfernt Zeilen ohne gültigen Schlusskurs, ergänzt fehlende Eröffnungs-/Hoch-/Tiefkurse
    * korrigiert unplausible OHLC-Kombinationen (Hoch < Schluss usw.)
    * erkennt Lücken, verdächtige Kurssprünge (> 15 % bei ±10 % BIST-Preisgrenze), Handelsaussetzungen,
      fehlendes Volumen (z. B. bei Indizes) und veraltete Daten
    """
    q = DatenQualitaet(zeilen_roh=0 if df is None else len(df))
    if df is None or df.empty:
        q.warnungen.append("Es liegen keine Kursdaten vor.")
        return pd.DataFrame(columns=OHLCV), q
    jetzt = jetzt if jetzt is not None else _jetzt_istanbul()

    d = df.copy()
    for spalte in OHLCV:
        if spalte not in d.columns:
            d[spalte] = np.nan
        d[spalte] = pd.to_numeric(d[spalte], errors="coerce").astype(float)
    d = d.replace([np.inf, -np.inf], np.nan)
    q.fehlende_werte = {s: int(n) for s in OHLCV if (n := d[s].isna().sum()) > 0}

    d = d[d["Close"].notna() & (d["Close"] > 0)]
    d = d[~d.index.duplicated(keep="last")].sort_index()
    d["Open"] = d["Open"].where(d["Open"] > 0).fillna(d["Close"])
    d["High"] = d["High"].where(d["High"] > 0).fillna(d[["Open", "Close"]].max(axis=1))
    d["Low"] = d["Low"].where(d["Low"] > 0).fillna(d[["Open", "Close"]].min(axis=1))
    hoch_neu = d[["High", "Open", "Close"]].max(axis=1)
    tief_neu = d[["Low", "Open", "Close"]].min(axis=1)
    q.korrigierte_ohlc = int(((hoch_neu != d["High"]) | (tief_neu != d["Low"])).sum())
    d["High"], d["Low"] = hoch_neu, tief_neu
    d["Volume"] = d["Volume"].fillna(0.0).clip(lower=0.0)

    q.zeilen_bereinigt = len(d)
    q.entfernte_zeilen = q.zeilen_roh - len(d)
    if d.empty:
        q.warnungen.append("Nach der Bereinigung sind keine gültigen Kurse übrig.")
        return d, q
    q.erste_kerze, q.letzte_kerze = d.index[0], d.index[-1]

    if q.entfernte_zeilen:
        q.hinweise.append(f"{q.entfernte_zeilen} Zeile(n) ohne gültigen Schlusskurs bzw. doppelte Zeitstempel "
                          "wurden entfernt.")
    if q.korrigierte_ohlc:
        q.hinweise.append(f"{q.korrigierte_ohlc} Kerze(n) mit unplausiblen Hoch-/Tiefkursen wurden korrigiert.")

    # Volumen verfügbar? (Indizes liefern je nach Anbieter kein Volumen)
    q.null_volumen_anteil = float((d["Volume"] <= 0).mean())
    q.volumen_verfuegbar = q.null_volumen_anteil < 0.5
    if not q.volumen_verfuegbar:
        q.hinweise.append("Keine verlässlichen Volumendaten – Volumen-Signale, OBV und Volumenbedingungen "
                          "werden nicht berücksichtigt.")

    # Lücken in der Zeitreihe (Feiertage wie Ramazan/Kurban Bayramı sind normal, größere Lücken nicht)
    if len(d) > 1:
        zeiten = d.index.to_series()
        abstand = zeiten.diff()
        grenze = pd.Timedelta(days=5) if intervall.intraday else pd.Timedelta(days=10)
        for ts in abstand[abstand > grenze].index:
            vorher = zeiten.shift(1).loc[ts]
            q.luecken.append({"von": vorher, "bis": ts, "tage": int(abstand.loc[ts].days)})
        if q.luecken:
            q.warnungen.append(f"{len(q.luecken)} größere Lücke(n) in der Zeitreihe (über {grenze.days} Tage) – "
                               "z. B. Handelsaussetzung oder fehlende Daten beim Anbieter.")

    # Verdächtige Kurssprünge (nur Tagesdaten)
    if not intervall.intraday and len(d) > 1:
        rendite = d["Close"].pct_change()
        maske = rendite.abs() > KURSSPRUNG_WARNSCHWELLE
        if maske.any():
            q.kurs_spruenge = pd.DataFrame({
                "Datum": d.index[maske],
                "Vortag": d["Close"].shift(1)[maske].to_numpy(),
                "Schluss": d["Close"][maske].to_numpy(),
                "Veränderung": rendite[maske].to_numpy(),
            })
            q.warnungen.append(
                f"{int(maske.sum())} ungewöhnliche Kurssprünge (über {KURSSPRUNG_WARNSCHWELLE:.0%} zum Vortag). "
                f"Da die meisten BIST-Aktien einer täglichen Preisgrenze von ±{BIST_PREISGRENZE:.0%} unterliegen, "
                "deutet das auf nicht bereinigte Kapitalmaßnahmen (z. B. Bonusaktien/Split) oder Datenfehler hin. "
                "Indikatoren, Signale und Backtest können in diesem Bereich verzerrt sein.")

    # Mögliche Handelsaussetzungen: ≥ 3 Kerzen ohne Umsatz und ohne Kursspanne
    if q.volumen_verfuegbar:
        still = (d["Volume"] <= 0) & (d["High"] == d["Low"])
        laengen = still.groupby((still != still.shift()).cumsum()).sum()
        q.handelsaussetzungen = int((laengen >= 3).sum())
        if q.handelsaussetzungen:
            q.warnungen.append(f"{q.handelsaussetzungen} Phase(n) ohne Handel erkannt (mögliche Handelsaussetzung).")

    # Veraltete Daten?
    if live:
        if intervall.intraday:
            veraltet = jetzt - d.index[-1] > pd.Timedelta(days=4)
        else:
            veraltet = len(pd.bdate_range(d.index[-1] + pd.Timedelta(days=1), jetzt.normalize())) > 5
        if veraltet:
            q.warnungen.append(f"Die letzte verfügbare Kerze stammt vom {fmt_datum(d.index[-1], intervall.intraday)}. "
                               "Die Daten könnten veraltet sein (Handelsaussetzung, Delisting oder Verzögerung).")

    extra = [s for s in ("Dividends", "Stock Splits") if s in d.columns]
    return d[OHLCV + extra], q


def letzte_kerze_unvollstaendig(index: pd.DatetimeIndex, intervall: IntervallInfo,
                                jetzt: pd.Timestamp | None = None) -> bool:
    """True, wenn die letzte Kerze noch nicht abgeschlossen ist (laufende Handelssitzung, BIST-Zeiten)."""
    if len(index) == 0:
        return False
    jetzt = jetzt if jetzt is not None else _jetzt_istanbul()
    letzte = pd.Timestamp(index[-1])
    sitzungsende = jetzt.normalize() + pd.Timedelta(minutes=BIST_SITZUNGSENDE_MIN + DATENPUFFER_MIN)
    if intervall.intraday:
        return letzte + pd.Timedelta(minutes=intervall.minuten or 60) > jetzt
    if intervall.code == "1d":
        return letzte.normalize() == jetzt.normalize() and jetzt < sitzungsende
    freitag = letzte.normalize() + pd.Timedelta(days=(4 - letzte.weekday()) % 7)   # Wochenkerze
    if jetzt.normalize() < freitag:
        return True
    return jetzt.normalize() == freitag and jetzt < sitzungsende


def auf_intervall_bringen(df: pd.DataFrame, intervall: IntervallInfo) -> pd.DataFrame:
    """Verdichtet Tagesdaten zu Wochenkerzen (Datum = letzter tatsächlicher Handelstag der Woche)."""
    if intervall.code != "1wk" or df.empty:
        return df
    hilfe = df[OHLCV].assign(_Datum=df.index)
    woche = (hilfe.groupby(pd.Grouper(freq="W-FRI"))
             .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum", "_Datum": "last"})
             .dropna(subset=["Close"]))
    woche.index = pd.DatetimeIndex(woche.pop("_Datum"))
    woche.index.name = "Datum"
    return woche


def wechselkurse_laden(ziel: str, start: pd.Timestamp, ende: pd.Timestamp) -> pd.Series:
    """Lädt Tages-Wechselkurse (TRY je Einheit der Zielwährung) von Yahoo Finance."""
    paar = WAEHRUNGEN.get(ziel)
    if not paar:
        raise DatenFehler(f"Für die Währung „{ziel}“ ist keine Umrechnung hinterlegt.", art="eingabe")
    fx, _ = _yahoo_laden(paar, start - pd.Timedelta(days=10), ende, "1d", dividendenbereinigt=False)
    return fx["Close"].rename(paar)


def waehrung_umrechnen(df: pd.DataFrame, wechselkurs: pd.Series) -> pd.DataFrame:
    """Rechnet TRY-Kurse in eine Fremdwährung um (Kurs / Wechselkurs des jeweiligen Tages).

    Es wird stets der letzte bekannte Wechselkurs verwendet (Vorwärtsauffüllung) – keine Zukunftswerte.
    Hoch/Tief werden mit dem Tagesschlusskurs des Wechselkurses umgerechnet (Näherung).
    """
    kurs = wechselkurs.dropna().sort_index()
    kurs = kurs[kurs > 0]
    kurs = kurs[~kurs.index.duplicated(keep="last")]
    tage = pd.DatetimeIndex(df.index.normalize())
    voll = kurs.reindex(kurs.index.union(tage.unique())).ffill()
    faktor = pd.Series(voll.reindex(tage).to_numpy(), index=df.index)
    d = df.copy()
    for spalte in ("Open", "High", "Low", "Close"):
        d[spalte] = d[spalte] / faktor
    return d.dropna(subset=["Close"])


def kursuebersicht_berechnen(daten: pd.DataFrame, intervall: IntervallInfo) -> dict[str, Any]:
    """Letzter Kurs, Tagesveränderung, Tagesvolumen, Umsatz und 52-Wochen-Spanne."""
    if intervall.intraday:
        tage = (daten.groupby(daten.index.normalize())
                .agg(Open=("Open", "first"), High=("High", "max"), Low=("Low", "min"),
                     Close=("Close", "last"), Volume=("Volume", "sum")))
    else:
        tage = daten
    kurs = float(daten["Close"].iloc[-1])
    vortag = float(tage["Close"].iloc[-2]) if len(tage) > 1 else float("nan")
    volumen = float(tage["Volume"].iloc[-1])
    abdeckung = (tage.index[-1] - tage.index[0]).days
    jahr = tage.loc[tage.index > tage.index[-1] - pd.Timedelta(days=365)]
    return {
        "kurs": kurs,
        "vortag": vortag,
        "veraenderung": kurs - vortag,
        "veraenderung_pct": kurs / vortag - 1 if vortag and math.isfinite(vortag) else float("nan"),
        "volumen": volumen,
        "umsatz": volumen * kurs,
        "tageshoch": float(tage["High"].iloc[-1]),
        "tagestief": float(tage["Low"].iloc[-1]),
        "datum": daten.index[-1],
        "hoch_52w": float(jahr["High"].max()) if abdeckung >= 300 else float("nan"),
        "tief_52w": float(jahr["Low"].min()) if abdeckung >= 300 else float("nan"),
    }


# =============================================================================
# 4) Technische Indikatoren (alle Berechnungen nutzen nur vergangene Werte)
# =============================================================================

def _exponentiell_glaetten(werte: pd.Series, periode: int, alpha: float) -> pd.Series:
    """Rekursive Glättung mit SMA-Startwert (wie TA-Lib).

    Startwert = Durchschnitt der ersten ``periode`` gültigen Werte, danach
    ``y_t = y_{t-1} + alpha · (x_t − y_{t-1})``. Fehlende Werte übernehmen den Vorwert.
    """
    arr = werte.to_numpy(dtype=float)
    ergebnis = np.full(arr.shape, np.nan)
    gueltig = np.flatnonzero(~np.isnan(arr))
    if gueltig.size < periode:
        return pd.Series(ergebnis, index=werte.index)
    start = int(gueltig[0])
    ende_start = start + periode
    if ende_start > arr.size:
        return pd.Series(ergebnis, index=werte.index)
    wert = float(np.nanmean(arr[start:ende_start]))
    ergebnis[ende_start - 1] = wert
    for i in range(ende_start, arr.size):
        x = arr[i]
        if not math.isnan(x):
            wert += alpha * (x - wert)
        ergebnis[i] = wert
    return pd.Series(ergebnis, index=werte.index)


def _wilder(werte: pd.Series, periode: int) -> pd.Series:
    """Wilder-Glättung (RMA) für RSI, ATR und ADX."""
    return _exponentiell_glaetten(werte, periode, 1.0 / periode)


def _ema(werte: pd.Series, periode: int) -> pd.Series:
    """Exponentieller gleitender Durchschnitt (Glättungsfaktor 2 / (n + 1))."""
    return _exponentiell_glaetten(werte, periode, 2.0 / (periode + 1))


def _rsi(schluss: pd.Series, periode: int = 14) -> pd.Series:
    """Relative Strength Index nach Wilder."""
    delta = schluss.diff()
    gewinn = _wilder(delta.clip(lower=0), periode)
    verlust = _wilder((-delta).clip(lower=0), periode)
    rs = gewinn / verlust.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.mask((verlust == 0) & (gewinn > 0), 100.0)
    rsi = rsi.mask((verlust == 0) & (gewinn == 0), 50.0)
    return rsi.where(gewinn.notna())


def indikatoren_berechnen(df: pd.DataFrame) -> pd.DataFrame:
    """Berechnet alle technischen Indikatoren.

    SMA 20/50/200, EMA 12/26, MACD (12/26/9), RSI 14, Stochastik (14/3/3), ATR 14, ADX 14 mit ±DI,
    OBV, Bollinger-Bänder (20/2), Volumen-Durchschnitt (20) und relative Volumenstärke.
    Alle Werte zum Zeitpunkt t hängen ausschließlich von Daten bis einschließlich t ab (kein Look-Ahead).
    """
    if df is None or df.empty:
        raise ValueError("Keine Kursdaten für die Indikatorberechnung vorhanden.")
    d = df.copy()
    schluss = d["Close"].astype(float)
    hoch = d["High"].astype(float)
    tief = d["Low"].astype(float)
    volumen = d["Volume"].astype(float).fillna(0.0) if "Volume" in d else pd.Series(0.0, index=d.index)

    # Gleitende Durchschnitte
    for n in (20, 50, 200):
        d[f"SMA_{n}"] = schluss.rolling(n, min_periods=n).mean()
    d["EMA_12"] = _ema(schluss, 12)
    d["EMA_26"] = _ema(schluss, 26)

    # MACD
    d["MACD"] = d["EMA_12"] - d["EMA_26"]
    d["MACD_Signal"] = _ema(d["MACD"], 9)
    d["MACD_Hist"] = d["MACD"] - d["MACD_Signal"]

    # RSI
    d["RSI_14"] = _rsi(schluss, 14)

    # Stochastik (langsam): %K = SMA3 des schnellen %K, %D = SMA3 von %K
    tief_14 = tief.rolling(14, min_periods=14).min()
    hoch_14 = hoch.rolling(14, min_periods=14).max()
    k_schnell = 100 * (schluss - tief_14) / (hoch_14 - tief_14).replace(0, np.nan)
    d["Stoch_K"] = k_schnell.rolling(3, min_periods=3).mean()
    d["Stoch_D"] = d["Stoch_K"].rolling(3, min_periods=3).mean()

    # ATR (Wilder)
    vorschluss = schluss.shift(1)
    wahre_spanne = pd.concat([hoch - tief, (hoch - vorschluss).abs(), (tief - vorschluss).abs()],
                             axis=1).max(axis=1, skipna=False)
    d["ATR_14"] = _wilder(wahre_spanne, 14)
    d["ATR_Prozent"] = d["ATR_14"] / schluss * 100

    # ADX mit +DI / −DI
    auf = hoch.diff()
    ab = -tief.diff()
    plus_dm = pd.Series(np.where((auf > ab) & (auf > 0), auf, 0.0), index=d.index)
    minus_dm = pd.Series(np.where((ab > auf) & (ab > 0), ab, 0.0), index=d.index)
    plus_dm.iloc[0] = np.nan
    minus_dm.iloc[0] = np.nan
    atr_basis = d["ATR_14"].replace(0, np.nan)
    d["Plus_DI"] = 100 * _wilder(plus_dm, 14) / atr_basis
    d["Minus_DI"] = 100 * _wilder(minus_dm, 14) / atr_basis
    dx = 100 * (d["Plus_DI"] - d["Minus_DI"]).abs() / (d["Plus_DI"] + d["Minus_DI"]).replace(0, np.nan)
    d["ADX_14"] = _wilder(dx, 14)

    # On-Balance-Volume
    richtung = np.sign(schluss.diff()).fillna(0.0)
    d["OBV"] = (richtung * volumen).cumsum()
    d["OBV_SMA_20"] = d["OBV"].rolling(20, min_periods=20).mean()

    # Bollinger-Bänder (20 Perioden, 2 Standardabweichungen)
    std_20 = schluss.rolling(20, min_periods=20).std(ddof=0)
    d["BB_Mitte"] = d["SMA_20"]
    d["BB_Oben"] = d["SMA_20"] + 2 * std_20
    d["BB_Unten"] = d["SMA_20"] - 2 * std_20
    bandbreite = d["BB_Oben"] - d["BB_Unten"]
    d["BB_ProzentB"] = (schluss - d["BB_Unten"]) / bandbreite.replace(0, np.nan)
    d["BB_Bandbreite"] = bandbreite / d["BB_Mitte"] * 100

    # Volumen
    d["Volumen_SMA_20"] = volumen.rolling(20, min_periods=20).mean()
    d["Rel_Volumen"] = volumen / d["Volumen_SMA_20"].replace(0, np.nan)

    d["Rendite"] = schluss.pct_change()
    return d


# =============================================================================
# 5) Trendanalyse
# =============================================================================

TREND_TEXT = {1: "aufwärtsgerichtet", 0: "seitwärts / unklar", -1: "abwärtsgerichtet"}


def trend_analyse(df: pd.DataFrame, sp: StrategieParameter | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Bestimmt kurz-, mittel- und langfristigen Trend sowie die Trendstärke (ADX).

    * kurzfristig: Kurs vs. SMA 20 und Steigung der SMA 20 (5 Kerzen)
    * mittelfristig (maßgeblich für Signale): Kurs > SMA 50, SMA 20 > SMA 50, SMA 50 steigt (10 Kerzen)
    * langfristig: Kurs und SMA 50 relativ zur SMA 200
    """
    sp = sp or StrategieParameter()
    d = df.copy()
    c = d["Close"]
    steigung_20 = d["SMA_20"] - d["SMA_20"].shift(5)
    steigung_50 = d["SMA_50"] - d["SMA_50"].shift(10)
    d["Trend_kurz"] = np.select(
        [(c > d["SMA_20"]) & (steigung_20 > 0), (c < d["SMA_20"]) & (steigung_20 < 0)], [1, -1], 0)
    d["Trend_mittel"] = np.select(
        [(c > d["SMA_50"]) & (d["SMA_20"] > d["SMA_50"]) & (steigung_50 > 0),
         (c < d["SMA_50"]) & (d["SMA_20"] < d["SMA_50"]) & (steigung_50 < 0)], [1, -1], 0)
    d["Trend_lang"] = np.select(
        [(c > d["SMA_200"]) & (d["SMA_50"] > d["SMA_200"]),
         (c < d["SMA_200"]) & (d["SMA_50"] < d["SMA_200"])], [1, -1], 0)
    d["Trend"] = d["Trend_mittel"]
    d["Trendstaerke"] = np.select([d["ADX_14"] >= sp.adx_schwelle, d["ADX_14"] >= sp.adx_schwelle - 5],
                                  ["stark", "moderat"], "schwach")

    z = d.iloc[-1]
    verfuegbar = {"kurz": pd.notna(z["SMA_20"]), "mittel": pd.notna(z["SMA_50"]), "lang": pd.notna(z["SMA_200"])}
    beschreibung = {
        "kurz": "Kurs vs. SMA 20 und deren Steigung",
        "mittel": "Kurs, SMA 20 und Steigung relativ zur SMA 50",
        "lang": "Kurs und SMA 50 relativ zur SMA 200",
    }
    horizonte = {}
    for schluessel, spalte in (("kurz", "Trend_kurz"), ("mittel", "Trend_mittel"), ("lang", "Trend_lang")):
        wert = int(z[spalte])
        horizonte[schluessel] = {
            "wert": wert if verfuegbar[schluessel] else None,
            "text": TREND_TEXT[wert] if verfuegbar[schluessel] else "nicht bestimmbar (zu wenige Kerzen)",
            "grundlage": beschreibung[schluessel],
        }
    adx = z.get("ADX_14", np.nan)
    if pd.notna(adx):
        richtung = ("Käufer dominieren (+DI > −DI)" if z["Plus_DI"] > z["Minus_DI"]
                    else "Verkäufer dominieren (−DI > +DI)")
        staerke = f"{z['Trendstaerke']} (ADX {fmt_zahl(adx, 1)})"
    else:
        richtung, staerke = "nicht bestimmbar", "nicht bestimmbar"
    mittel = horizonte["mittel"]["text"]
    fazit = (f"Mittelfristiger Trend: {mittel}; Trendstärke: {staerke}. "
             f"Langfristig: {horizonte['lang']['text']}.")
    return d, {"horizonte": horizonte, "staerke": staerke, "richtung": richtung, "fazit": fazit}


# =============================================================================
# 6) Punktesystem
# =============================================================================

def _ereignis_im_fenster(ereignis: pd.Series, fenster: int) -> pd.Series:
    """True, wenn das Ereignis in den letzten ``fenster`` Kerzen (inkl. aktueller) aufgetreten ist."""
    return (ereignis.astype(float).rolling(max(int(fenster), 1), min_periods=1).max()
            .fillna(0).astype(bool))


def bewertung_einordnen(punkte: float, sp: StrategieParameter) -> tuple[str, str]:
    """Ordnet eine Punktzahl anhand der (änderbaren) Schwellenwerte ein → (Text, Stil)."""
    if punkte >= sp.schwelle_stark_positiv:
        return "Stark positives Signal", "stark_pos"
    if punkte >= sp.schwelle_positiv:
        return "Positives Signal", "pos"
    if punkte <= sp.schwelle_stark_negativ:
        return "Stark negatives Signal", "stark_neg"
    if punkte <= sp.schwelle_negativ:
        return "Negatives Signal", "neg"
    return "Neutral", "neutral"


def signalpunkte_berechnen(df: pd.DataFrame, sp: StrategieParameter | None = None,
                           volumen_verfuegbar: bool = True) -> pd.DataFrame:
    """Berechnet je Kerze die Punkte der Bewertungskomponenten und die Gesamtpunktzahl.

    Kreuzungen (Golden/Death Cross, MACD) sind Ereignisse: Sie zählen innerhalb eines einstellbaren
    Fensters, solange die Kreuzung nicht wieder aufgehoben wurde. Das Volumen-Signal zählt im
    MACD-Fenster. Fehlende Indikatorwerte (z. B. SMA 200 in den ersten 199 Kerzen) ergeben 0 Punkte.
    """
    sp = sp or StrategieParameter()
    g = sp.gewichte
    d = df.copy()
    c, s50, s200 = d["Close"], d["SMA_50"], d["SMA_200"]
    macd, signal, rsi = d["MACD"], d["MACD_Signal"], d["RSI_14"]

    d["Golden_Cross"] = (s50 > s200) & (s50.shift(1) <= s200.shift(1))
    d["Death_Cross"] = (s50 < s200) & (s50.shift(1) >= s200.shift(1))
    d["MACD_Kreuz_hoch"] = (macd > signal) & (macd.shift(1) <= signal.shift(1))
    d["MACD_Kreuz_runter"] = (macd < signal) & (macd.shift(1) >= signal.shift(1))
    golden_aktiv = _ereignis_im_fenster(d["Golden_Cross"], sp.sma_kreuz_fenster) & (s50 > s200)
    death_aktiv = _ereignis_im_fenster(d["Death_Cross"], sp.sma_kreuz_fenster) & (s50 < s200)
    d["MACD_Kaufimpuls"] = _ereignis_im_fenster(d["MACD_Kreuz_hoch"], sp.macd_fenster) & (macd > signal)
    d["MACD_Verkaufsimpuls"] = _ereignis_im_fenster(d["MACD_Kreuz_runter"], sp.macd_fenster) & (macd < signal)

    d["P_SMA_Kreuz"] = np.select([golden_aktiv, death_aktiv], [g.golden_cross, g.death_cross], 0)
    d["P_RSI"] = np.select(
        [rsi < sp.rsi_ueberverkauft, (rsi >= sp.rsi_ueberverkauft) & (rsi <= 50), rsi > sp.rsi_ueberkauft],
        [g.rsi_ueberverkauft, g.rsi_30_bis_50, g.rsi_ueberkauft], 0)
    d["P_MACD"] = np.select([d["MACD_Kaufimpuls"], d["MACD_Verkaufsimpuls"]],
                            [g.macd_kreuz_hoch, g.macd_kreuz_runter], 0)
    d["P_SMA200"] = np.select([c > s200, c < s200], [g.kurs_ueber_sma200, g.kurs_unter_sma200], 0)
    if volumen_verfuegbar:
        # Volumen-Signal: erhöhtes Volumen an einem steigenden (+) bzw. fallenden (−) Tag. Es zählt – wie die
        # MACD-Kreuzung – innerhalb des Signalfensters (jüngstes Volumen-Signal gilt).
        erhoeht = d["Rel_Volumen"] >= sp.volumen_faktor
        roh = pd.Series(np.select([erhoeht & (c > c.shift(1)), erhoeht & (c < c.shift(1))], [1, -1], 0),
                        index=d.index)
        d["Volumen_Signal"] = roh
        fenster = max(int(sp.macd_fenster), 1)
        aktiv = roh.replace(0, np.nan).ffill(limit=fenster - 1).fillna(0) if fenster > 1 else roh
        d["P_Volumen"] = np.select([aktiv > 0, aktiv < 0], [g.volumen_positiv, g.volumen_negativ], 0)
    else:
        d["Volumen_Signal"] = 0
        d["P_Volumen"] = 0
    adx_stark = d["ADX_14"] > sp.adx_schwelle
    d["P_ADX"] = np.select([adx_stark & (d["Plus_DI"] > d["Minus_DI"]), adx_stark & (d["Minus_DI"] > d["Plus_DI"])],
                           [g.adx_trend_positiv, g.adx_trend_negativ], 0)
    d["Punkte"] = d[list(PUNKTE_SPALTEN)].sum(axis=1).astype(int)
    d["Bewertung"] = [bewertung_einordnen(p, sp)[0] for p in d["Punkte"]]
    return d


def punkte_aufschluesseln(z: pd.Series, sp: StrategieParameter, volumen_verfuegbar: bool = True) -> pd.DataFrame:
    """Erklärt die Punkte der letzten Kerze je Komponente (für die Signalbegründung)."""
    def zahl(spalte: str, stellen: int = 2) -> str:
        return fmt_zahl(z.get(spalte), stellen)

    rsi = z.get("RSI_14")
    zeilen = []
    # Golden/Death Cross
    if pd.isna(z.get("SMA_200")):
        text_kreuz = "SMA 200 noch nicht verfügbar"
    elif z["P_SMA_Kreuz"] > 0:
        text_kreuz = f"Golden Cross innerhalb der letzten {sp.sma_kreuz_fenster} Kerzen"
    elif z["P_SMA_Kreuz"] < 0:
        text_kreuz = f"Death Cross innerhalb der letzten {sp.sma_kreuz_fenster} Kerzen"
    else:
        lage = "über" if z["SMA_50"] > z["SMA_200"] else "unter"
        text_kreuz = f"keine frische Kreuzung (SMA 50 liegt {lage} SMA 200)"
    zeilen.append(("P_SMA_Kreuz", f"SMA 50: {zahl('SMA_50')} / SMA 200: {zahl('SMA_200')}", text_kreuz))
    # RSI
    if pd.isna(rsi):
        text_rsi = "RSI noch nicht verfügbar"
    elif rsi < sp.rsi_ueberverkauft:
        text_rsi = f"überverkauft (unter {sp.rsi_ueberverkauft:g})"
    elif rsi <= 50:
        text_rsi = f"zwischen {sp.rsi_ueberverkauft:g} und 50 (Erholungspotenzial)"
    elif rsi > sp.rsi_ueberkauft:
        text_rsi = f"überkauft (über {sp.rsi_ueberkauft:g})"
    else:
        text_rsi = f"zwischen 50 und {sp.rsi_ueberkauft:g} (neutral)"
    zeilen.append(("P_RSI", fmt_zahl(rsi, 1), text_rsi))
    # MACD
    if z["P_MACD"] > 0:
        text_macd = f"Kreuzung über die Signallinie (innerhalb {sp.macd_fenster} Kerzen)"
    elif z["P_MACD"] < 0:
        text_macd = f"Kreuzung unter die Signallinie (innerhalb {sp.macd_fenster} Kerzen)"
    else:
        lage = "über" if z.get("MACD", 0) > z.get("MACD_Signal", 0) else "unter"
        text_macd = f"keine frische Kreuzung (MACD {lage} Signallinie)"
    zeilen.append(("P_MACD", f"MACD {zahl('MACD', 3)} / Signal {zahl('MACD_Signal', 3)}", text_macd))
    # Kurs vs. SMA 200
    if pd.isna(z.get("SMA_200")):
        text_200 = "SMA 200 noch nicht verfügbar"
    else:
        text_200 = "Schlusskurs über SMA 200" if z["Close"] > z["SMA_200"] else "Schlusskurs unter SMA 200"
    zeilen.append(("P_SMA200", f"Kurs {zahl('Close')} / SMA 200 {zahl('SMA_200')}", text_200))
    # Volumen
    if not volumen_verfuegbar:
        text_vol, wert_vol = "keine Volumendaten", "–"
    else:
        wert_vol = f"{fmt_zahl(z.get('Rel_Volumen'), 2)} × Ø"
        zeitraum = "auf der letzten Kerze" if sp.macd_fenster <= 1 else f"in den letzten {sp.macd_fenster} Kerzen"
        if z["P_Volumen"] > 0:
            text_vol = f"erhöhtes Volumen (≥ {fmt_zahl(sp.volumen_faktor, 1)} × Ø) bei steigendem Kurs {zeitraum}"
        elif z["P_Volumen"] < 0:
            text_vol = f"erhöhtes Volumen (≥ {fmt_zahl(sp.volumen_faktor, 1)} × Ø) bei fallendem Kurs {zeitraum}"
        else:
            text_vol = f"kein auffälliges Volumen {zeitraum}"
    zeilen.append(("P_Volumen", wert_vol, text_vol))
    # ADX
    if pd.isna(z.get("ADX_14")):
        text_adx = "ADX noch nicht verfügbar"
    elif z["P_ADX"] > 0:
        text_adx = f"ADX über {sp.adx_schwelle:g} bei positivem Trend (+DI > −DI)"
    elif z["P_ADX"] < 0:
        text_adx = f"ADX über {sp.adx_schwelle:g} bei negativem Trend (−DI > +DI)"
    else:
        text_adx = f"ADX unter {sp.adx_schwelle:g} – kein ausgeprägter Trend"
    zeilen.append(("P_ADX", f"ADX {zahl('ADX_14', 1)} (+DI {zahl('Plus_DI', 1)} / −DI {zahl('Minus_DI', 1)})",
                   text_adx))
    return pd.DataFrame([{"Komponente": PUNKTE_SPALTEN[s], "Wert": w, "Punkte": int(z[s]), "Begründung": t}
                         for s, w, t in zeilen])


# =============================================================================
# 7) Kauf- und Verkaufssignale
# =============================================================================

def _divergenzen(d: pd.DataFrame, fenster: int = 14) -> tuple[pd.Series, pd.Series]:
    """Einfache RSI-Divergenzen ohne Zukunftswerte.

    Negativ: Kurs erreicht in den letzten ``fenster`` Kerzen ein höheres Hoch als im Fenster davor,
    das RSI-Hoch fällt jedoch niedriger aus (vorheriges RSI-Hoch > 60). Positiv entsprechend umgekehrt.
    """
    hoch_jetzt = d["High"].rolling(fenster, min_periods=fenster).max()
    hoch_vorher = d["High"].shift(fenster).rolling(fenster, min_periods=fenster).max()
    rsi_hoch_jetzt = d["RSI_14"].rolling(fenster, min_periods=fenster).max()
    rsi_hoch_vorher = d["RSI_14"].shift(fenster).rolling(fenster, min_periods=fenster).max()
    negativ = (hoch_jetzt > hoch_vorher) & (rsi_hoch_jetzt < rsi_hoch_vorher - 2) & (rsi_hoch_vorher > 60)

    tief_jetzt = d["Low"].rolling(fenster, min_periods=fenster).min()
    tief_vorher = d["Low"].shift(fenster).rolling(fenster, min_periods=fenster).min()
    rsi_tief_jetzt = d["RSI_14"].rolling(fenster, min_periods=fenster).min()
    rsi_tief_vorher = d["RSI_14"].shift(fenster).rolling(fenster, min_periods=fenster).min()
    positiv = (tief_jetzt < tief_vorher) & (rsi_tief_jetzt > rsi_tief_vorher + 2) & (rsi_tief_vorher < 40)
    return negativ.fillna(False), positiv.fillna(False)


def signalbedingungen(d: pd.DataFrame, sp: StrategieParameter,
                      volumen_verfuegbar: bool = True) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    """Liefert die Bestätigungsbedingungen für Kauf- und Verkaufssignale (je Kerze True/False)."""
    c, rsi, adx = d["Close"], d["RSI_14"], d["ADX_14"]
    plus_di, minus_di, rel_vol = d["Plus_DI"], d["Minus_DI"], d["Rel_Volumen"]
    kauf = {
        "Aufwärtstrend": d["Trend"] == 1,
        "Kurs über SMA 200": c > d["SMA_200"],
        "SMA 50 über SMA 200": d["SMA_50"] > d["SMA_200"],
        f"RSI zwischen {sp.rsi_ueberverkauft:g} und {sp.rsi_kauf_obergrenze:g}":
            (rsi >= sp.rsi_ueberverkauft) & (rsi <= sp.rsi_kauf_obergrenze),
        "Volumen über Durchschnitt": rel_vol >= sp.volumen_faktor,
        "ADX bestätigt Aufwärtstrend": (adx >= sp.adx_schwelle) & (plus_di > minus_di),
    }
    verkauf = {
        "Abwärtstrend": d["Trend"] == -1,
        "Kurs unter SMA 200": c < d["SMA_200"],
        "SMA 50 unter SMA 200": d["SMA_50"] < d["SMA_200"],
        f"RSI über {sp.rsi_ueberkauft:g} oder negative Divergenz": (rsi > sp.rsi_ueberkauft) | d["Neg_Divergenz"],
        "Steigendes Volumen bei fallenden Kursen":
            (c < c.shift(1)) & (rel_vol >= sp.volumen_faktor) & (d["Volume"] > d["Volume"].shift(1)),
        "ADX bestätigt Abwärtstrend": (adx >= sp.adx_schwelle) & (minus_di > plus_di),
    }
    if not volumen_verfuegbar:
        kauf.pop("Volumen über Durchschnitt")
        verkauf.pop("Steigendes Volumen bei fallenden Kursen")
    return ({k: v.fillna(False).astype(bool) for k, v in kauf.items()},
            {k: v.fillna(False).astype(bool) for k, v in verkauf.items()})


def _erstes_signal_je_ereignis(kandidat: pd.Series, ereignis: pd.Series) -> pd.Series:
    """Höchstens ein Signal pro MACD-Kreuzung (erste Kerze, an der alle Bedingungen erfüllt sind)."""
    gruppe = ereignis.astype(int).cumsum()
    laufend = kandidat.astype(int).groupby(gruppe).cumsum()
    return kandidat & (laufend == 1)


def kauf_und_verkaufssignale_ermitteln(df: pd.DataFrame, sp: StrategieParameter | None = None,
                                       volumen_verfuegbar: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Erzeugt Kauf-/Verkaufssignale nur, wenn mehrere Bedingungen gleichzeitig erfüllt sind.

    Kaufsignal: MACD kreuzt die Signallinie nach oben (Auslöser, zählt ``macd_fenster`` Kerzen)
    UND mindestens ``min_bestaetigungen`` Bestätigungen (Trend, SMA 200, SMA 50/200, RSI-Bereich,
    Volumen, ADX) UND Gesamtpunktzahl ≥ Schwelle „positiv“. Stark ab Schwelle „stark positiv“.
    Verkaufssignal spiegelbildlich. Alle Bedingungen verwenden ausschließlich Daten bis zur Signalkerze.
    """
    sp = sp or StrategieParameter()
    d = df.copy()
    d["Neg_Divergenz"], d["Pos_Divergenz"] = _divergenzen(d)
    kauf_bed, verkauf_bed = signalbedingungen(d, sp, volumen_verfuegbar)
    d["Kauf_Bestaetigungen"] = sum(s.astype(int) for s in kauf_bed.values())
    d["Verkauf_Bestaetigungen"] = sum(s.astype(int) for s in verkauf_bed.values())
    min_kauf = min(sp.min_bestaetigungen, len(kauf_bed))
    min_verkauf = min(sp.min_bestaetigungen, len(verkauf_bed))

    kauf_kandidat = (d["MACD_Kaufimpuls"] & (d["Kauf_Bestaetigungen"] >= min_kauf)
                     & (d["Punkte"] >= sp.schwelle_positiv))
    verkauf_kandidat = (d["MACD_Verkaufsimpuls"] & (d["Verkauf_Bestaetigungen"] >= min_verkauf)
                        & (d["Punkte"] <= sp.schwelle_negativ))
    kauf = _erstes_signal_je_ereignis(kauf_kandidat, d["MACD_Kreuz_hoch"])
    verkauf = _erstes_signal_je_ereignis(verkauf_kandidat, d["MACD_Kreuz_runter"])

    code = np.zeros(len(d), dtype=int)
    code[kauf.to_numpy()] = np.where(d.loc[kauf, "Punkte"] >= sp.schwelle_stark_positiv, 2, 1)
    code[verkauf.to_numpy()] = np.where(d.loc[verkauf, "Punkte"] <= sp.schwelle_stark_negativ, -2, -1)
    d["Signal"] = code
    d["Signaltyp"] = [SIGNAL_CODES[int(x)] for x in code]

    zeilen = []
    for ts in d.index[d["Signal"] != 0]:
        z = d.loc[ts]
        ist_kauf = z["Signal"] > 0
        bedingungen = kauf_bed if ist_kauf else verkauf_bed
        erfuellt = [name for name, reihe in bedingungen.items() if bool(reihe.loc[ts])]
        offen = [name for name in bedingungen if name not in erfuellt]
        gesamt = len(bedingungen) + 1
        anteil = (len(erfuellt) + 1) / gesamt
        ausloeser = "MACD kreuzt Signallinie " + ("nach oben" if ist_kauf else "nach unten")
        bewertung = bewertung_einordnen(z["Punkte"], sp)[0]
        erklaerung = (f"{ausloeser}; bestätigt durch {len(erfuellt)} von {len(bedingungen)} Bedingungen. "
                      f"RSI {fmt_zahl(z['RSI_14'], 0)}, ADX {fmt_zahl(z['ADX_14'], 0)}"
                      + (f", Volumen {fmt_zahl(z['Rel_Volumen'], 1)} × Ø" if volumen_verfuegbar else "")
                      + f". Gesamtpunktzahl {punkte_text(z['Punkte'])} ({bewertung}).")
        if offen:
            erklaerung += " Nicht erfüllt: " + ", ".join(offen) + "."
        zeilen.append({
            "Datum": ts,
            "Kurs": float(z["Close"]),
            "Signaltyp": z["Signaltyp"],
            "Punktzahl": int(z["Punkte"]),
            "Auslösende Indikatoren": "; ".join([ausloeser, *erfuellt]),
            "Signalstärke": f"{fmt_pct(anteil, 0, False)} ({len(erfuellt) + 1}/{gesamt} Bedingungen)",
            "Stärke": anteil,
            "Erklärung": erklaerung,
            "Code": int(z["Signal"]),
        })
    spalten = ["Datum", "Kurs", "Signaltyp", "Punktzahl", "Auslösende Indikatoren", "Signalstärke",
               "Stärke", "Erklärung", "Code"]
    return d, pd.DataFrame(zeilen, columns=spalten)


def aktueller_signalstatus(d: pd.DataFrame, sp: StrategieParameter) -> dict[str, Any]:
    """Aktueller Signalstatus: frisches Signal (innerhalb des MACD-Fensters) oder Halten/Beobachten."""
    fenster = max(int(sp.macd_fenster), 1)
    juengste = d["Signal"].iloc[-fenster:]
    ereignisse = juengste[juengste != 0]
    if not ereignisse.empty:
        ts = ereignisse.index[-1]
        code = int(ereignisse.iloc[-1])
        alter = len(d) - 1 - d.index.get_loc(ts)
        return {"status": SIGNAL_CODES[code], "code": code, "datum": ts, "alter": int(alter)}
    return {"status": HALTEN, "code": 0, "datum": None, "alter": None}


# =============================================================================
# 8) Unterstützung/Widerstand und Risikoanalyse
# =============================================================================

@dataclass
class Zone:
    """Unterstützungs- bzw. Widerstandszone aus gebündelten Umkehrpunkten."""

    unten: float
    oben: float
    art: str                  # "Unterstützung" oder "Widerstand"
    beruehrungen: int
    letzte_beruehrung: pd.Timestamp

    @property
    def mitte(self) -> float:
        return (self.unten + self.oben) / 2


def unterstuetzung_widerstand_ermitteln(d: pd.DataFrame, fenster: int = 5, max_je_seite: int = 2) -> list[Zone]:
    """Ermittelt Zonen aus lokalen Hoch- und Tiefpunkten (Pivots), die nah beieinander liegen.

    Ein Pivot gilt erst als bestätigt, wenn ``fenster`` Kerzen danach vorliegen. Die Zonen dienen
    ausschließlich der Darstellung und Risikoeinschätzung – sie fließen nicht in Signale/Backtest ein.
    """
    if len(d) < 2 * fenster + 5:
        return []
    breite = 2 * fenster + 1
    hoch, tief = d["High"], d["Low"]
    pivots_hoch = hoch[hoch == hoch.rolling(breite, center=True).max()]
    pivots_tief = tief[tief == tief.rolling(breite, center=True).min()]
    punkte = pd.concat([pivots_hoch, pivots_tief]).sort_values()
    if punkte.empty:
        return []
    kurs = float(d["Close"].iloc[-1])
    atr = d["ATR_14"].iloc[-1] if "ATR_14" in d and pd.notna(d["ATR_14"].iloc[-1]) \
        else float((hoch - tief).tail(14).mean())
    toleranz = max(0.8 * float(atr), 0.01 * kurs)

    gruppen: list[dict[str, Any]] = []
    for ts, preis in punkte.items():
        if gruppen and preis - gruppen[-1]["start"] <= toleranz:
            gruppen[-1]["preise"].append(float(preis))
            gruppen[-1]["zeiten"].append(ts)
        else:
            gruppen.append({"start": float(preis), "preise": [float(preis)], "zeiten": [ts]})

    zonen = []
    for gruppe in gruppen:
        unten, oben = min(gruppe["preise"]), max(gruppe["preise"])
        mindestbreite = 0.4 * toleranz
        if oben - unten < mindestbreite:
            polster = (mindestbreite - (oben - unten)) / 2
            unten, oben = unten - polster, oben + polster
        art = "Unterstützung" if (unten + oben) / 2 < kurs else "Widerstand"
        zonen.append(Zone(unten, oben, art, len(gruppe["preise"]), max(gruppe["zeiten"])))
    relevant = [z for z in zonen if z.beruehrungen >= 2] or zonen
    unterstuetzungen = sorted((z for z in relevant if z.art == "Unterstützung"), key=lambda z: kurs - z.oben)
    widerstaende = sorted((z for z in relevant if z.art == "Widerstand"), key=lambda z: z.unten - kurs)
    return unterstuetzungen[:max_je_seite] + widerstaende[:max_je_seite]


@dataclass
class RisikoErgebnis:
    """Kennzahlen der Risikoanalyse."""

    kurs: float
    atr: float
    atr_prozent: float
    stop_loss: float
    stop_abstand: float
    kursziel: float
    ziel_abstand: float
    risiko_je_aktie: float
    stueckzahl: int
    positionswert: float
    positionsanteil: float
    max_verlust: float
    volatilitaet_pa: float
    volatilitaet_20_pa: float
    var_95: float
    verlustserie_laenge: int
    verlustserie_verlust: float
    verlustserie_start: pd.Timestamp | None
    verlustserie_ende: pd.Timestamp | None
    max_drawdown: float
    dd_hoch: pd.Timestamp | None
    dd_tief: pd.Timestamp | None
    dd_erholung: pd.Timestamp | None
    dd_dauer_kerzen: int
    dd_dauer_tage: int
    sharpe: float | None
    sharpe_hinweis: str
    perioden_pro_jahr: float
    naechste_unterstuetzung: Zone | None
    naechster_widerstand: Zone | None
    hinweise: list[str] = field(default_factory=list)


def perioden_pro_jahr_schaetzen(index: pd.DatetimeIndex, intervall: IntervallInfo) -> float:
    """Anzahl Kerzen pro Jahr (für Annualisierung). Intraday: Kerzen je Handelstag × 252."""
    if not intervall.intraday or len(index) == 0:
        return intervall.perioden_pro_jahr
    je_tag = pd.Series(1, index=index).groupby(index.normalize()).sum()
    return float(je_tag.median() * HANDELSTAGE_PRO_JAHR) if len(je_tag) else intervall.perioden_pro_jahr


def sharpe_ratio(renditen: pd.Series, perioden_pro_jahr: float, risikofrei_pa: float = 0.0) -> tuple[float | None, str]:
    """Annualisierte Sharpe Ratio; None, wenn sie nicht sinnvoll berechenbar ist."""
    r = renditen.replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 30:
        return None, "Nicht sinnvoll berechenbar: mindestens 30 Renditen erforderlich."
    rf = (1 + risikofrei_pa) ** (1 / perioden_pro_jahr) - 1
    ueberschuss = r - rf
    std = float(ueberschuss.std(ddof=1))
    if not math.isfinite(std) or std == 0:
        return None, "Nicht sinnvoll berechenbar: keine Schwankung der Renditen."
    return float(ueberschuss.mean() / std * math.sqrt(perioden_pro_jahr)), ""


def _laengste_verlustserie(renditen: pd.Series) -> tuple[int, float, pd.Timestamp | None, pd.Timestamp | None]:
    """Längste Folge negativer Renditen (bei Gleichstand die verlustreichere)."""
    beste: tuple[int, float, Any, Any] = (0, 0.0, None, None)
    laenge, kumuliert, start = 0, 1.0, None
    for ts, r in renditen.dropna().items():
        if r < 0:
            if laenge == 0:
                start, kumuliert = ts, 1.0
            laenge += 1
            kumuliert *= 1 + r
            if laenge > beste[0] or (laenge == beste[0] and kumuliert - 1 < beste[1]):
                beste = (laenge, kumuliert - 1, start, ts)
        else:
            laenge = 0
    return beste


def drawdown_phase(werte: pd.Series) -> dict[str, Any]:
    """Maximaler Drawdown mit Hochpunkt, Tiefpunkt, Erholung und Dauer."""
    werte = werte.dropna()
    leer = {"max_dd": 0.0, "hoch": None, "tief": None, "erholung": None, "dauer_kerzen": 0, "dauer_tage": 0}
    if len(werte) < 2:
        return leer
    drawdown = werte / werte.cummax() - 1
    max_dd = float(drawdown.min())
    if max_dd >= 0:
        return leer
    tief_ts = drawdown.idxmin()
    hoch_ts = werte.loc[:tief_ts].idxmax()
    danach = werte.loc[tief_ts:]
    erholt = danach >= werte.loc[hoch_ts]
    erholung_ts = danach.index[int(np.argmax(erholt.to_numpy()))] if erholt.any() else None
    ende = erholung_ts if erholung_ts is not None else werte.index[-1]
    return {"max_dd": max_dd, "hoch": hoch_ts, "tief": tief_ts, "erholung": erholung_ts,
            "dauer_kerzen": int(werte.index.get_loc(ende) - werte.index.get_loc(hoch_ts)),
            "dauer_tage": int((ende - hoch_ts).days)}


def positionsgroesse_berechnen(kapital: float, risiko_pct: float, einstieg: float, stop: float,
                               kosten: float, slippage: float) -> tuple[int, float]:
    """Stückzahl, sodass der Verlust bis zum Stop (inkl. Kosten) das Risikobudget nicht übersteigt."""
    if not (_ist_zahl(einstieg) and _ist_zahl(stop)) or einstieg <= stop or einstieg <= 0:
        return 0, float("nan")
    risiko_je_aktie = (einstieg - stop) + einstieg * (kosten + slippage) + stop * (kosten + slippage)
    budget = kapital * risiko_pct / 100
    stueck_risiko = math.floor(budget / risiko_je_aktie) if risiko_je_aktie > 0 else 0
    stueck_kapital = math.floor(kapital / (einstieg * (1 + kosten + slippage)))
    return max(0, min(stueck_risiko, stueck_kapital)), risiko_je_aktie


def risiko_berechnen(d: pd.DataFrame, rp: RisikoParameter, intervall: IntervallInfo,
                     zonen: list[Zone] | None = None) -> RisikoErgebnis:
    """Risikoanalyse für den Anzeigezeitraum (Long-Position).

    Historische Volatilität, ATR-Stop-Loss (Kurs − Multiplikator × ATR), Kursziel über das
    Chance-Risiko-Verhältnis, risikobasierte Positionsgröße, VaR, Verlustserie, Drawdown und Sharpe Ratio.
    """
    hinweise: list[str] = []
    kurs = float(d["Close"].iloc[-1])
    atr = float(d["ATR_14"].iloc[-1]) if pd.notna(d["ATR_14"].iloc[-1]) else float("nan")
    kosten = rp.transaktionskosten_pct / 100
    slip = rp.slippage_pct / 100
    if math.isfinite(atr):
        stop = kurs - rp.atr_multiplikator * atr
        if stop <= 0:
            hinweise.append("Der ATR-Stop läge unter null – Multiplikator verringern.")
            stop = float("nan")
    else:
        stop = float("nan")
        hinweise.append("ATR noch nicht verfügbar (mindestens 15 Kerzen erforderlich).")
    ziel = kurs + rp.crv * (kurs - stop) if math.isfinite(stop) else float("nan")
    stueck, risiko_je_aktie = positionsgroesse_berechnen(rp.kapital, rp.risiko_pro_position_pct, kurs, stop,
                                                         kosten, slip)
    positionswert = stueck * kurs

    ppy = perioden_pro_jahr_schaetzen(d.index, intervall)
    renditen = d["Close"].pct_change()
    log_renditen = np.log(d["Close"]).diff()
    vola = float(log_renditen.std(ddof=1) * math.sqrt(ppy)) if log_renditen.count() > 2 else float("nan")
    vola_20 = (float(log_renditen.tail(20).std(ddof=1) * math.sqrt(ppy))
               if log_renditen.tail(20).count() > 2 else float("nan"))
    var_95 = float(-np.nanpercentile(renditen.dropna(), 5)) if renditen.count() >= 20 else float("nan")
    serie = _laengste_verlustserie(renditen)
    dd = drawdown_phase(d["Close"])
    sharpe, sharpe_hinweis = sharpe_ratio(renditen, ppy, rp.risikofreier_zins_pct / 100)

    zonen = zonen or []
    unter = [z for z in zonen if z.art == "Unterstützung"]
    ueber = [z for z in zonen if z.art == "Widerstand"]
    naechste_u = max(unter, key=lambda z: z.oben) if unter else None
    naechster_w = min(ueber, key=lambda z: z.unten) if ueber else None
    if naechster_w and math.isfinite(ziel) and ziel > naechster_w.unten:
        hinweise.append(f"Das Kursziel ({fmt_zahl(ziel)}) liegt über dem nächsten Widerstand "
                        f"({fmt_zahl(naechster_w.unten)}–{fmt_zahl(naechster_w.oben)}). Das Erreichen ist "
                        "technisch schwieriger.")
    if (naechste_u and math.isfinite(stop) and math.isfinite(atr)
            and naechste_u.unten <= stop <= naechste_u.oben + 0.5 * atr):
        hinweise.append(f"Der ATR-Stop ({fmt_zahl(stop)}) liegt in bzw. knapp über der Unterstützungszone "
                        f"{fmt_zahl(naechste_u.unten)}–{fmt_zahl(naechste_u.oben)}. Ein Stop knapp unterhalb "
                        "der Zone ist oft robuster (größerer Abstand → kleinere Positionsgröße).")
    if stueck == 0 and math.isfinite(stop):
        hinweise.append("Mit dem gewählten Kapital und Risiko ist nicht einmal eine Aktie möglich.")
    return RisikoErgebnis(
        kurs=kurs, atr=atr, atr_prozent=atr / kurs if math.isfinite(atr) else float("nan"),
        stop_loss=stop, stop_abstand=(stop / kurs - 1) if math.isfinite(stop) else float("nan"),
        kursziel=ziel, ziel_abstand=(ziel / kurs - 1) if math.isfinite(ziel) else float("nan"),
        risiko_je_aktie=risiko_je_aktie, stueckzahl=stueck, positionswert=positionswert,
        positionsanteil=positionswert / rp.kapital if rp.kapital else float("nan"),
        max_verlust=stueck * risiko_je_aktie if stueck else 0.0,
        volatilitaet_pa=vola, volatilitaet_20_pa=vola_20, var_95=var_95,
        verlustserie_laenge=serie[0], verlustserie_verlust=serie[1],
        verlustserie_start=serie[2], verlustserie_ende=serie[3],
        max_drawdown=dd["max_dd"], dd_hoch=dd["hoch"], dd_tief=dd["tief"], dd_erholung=dd["erholung"],
        dd_dauer_kerzen=dd["dauer_kerzen"], dd_dauer_tage=dd["dauer_tage"],
        sharpe=sharpe, sharpe_hinweis=sharpe_hinweis, perioden_pro_jahr=ppy,
        naechste_unterstuetzung=naechste_u, naechster_widerstand=naechster_w, hinweise=hinweise,
    )


# =============================================================================
# 9) Backtest (ohne Look-Ahead-Bias)
# =============================================================================

@dataclass
class BacktestErgebnis:
    """Ergebnis von backtest_durchfuehren()."""

    verlauf: pd.DataFrame              # Kapitalentwicklung Strategie und Buy & Hold je Kerze
    trades: pd.DataFrame               # Einzelne Trades
    kennzahlen: pd.DataFrame           # Formatierte Kennzahlen (Strategie vs. Buy & Hold)
    werte: dict[str, dict[str, Any]]   # Rohwerte der Kennzahlen
    hinweise: list[str] = field(default_factory=list)


def _kennzahlen_berechnen(kapitalkurve: pd.Series, startkapital: float, ppy: float, rf_pa: float,
                          trades: pd.DataFrame | None = None, position: pd.Series | None = None) -> dict[str, Any]:
    """Rendite- und Risikokennzahlen einer Kapitalkurve (optional inkl. Trade-Statistik)."""
    kurve = kapitalkurve.astype(float)
    endwert = float(kurve.iloc[-1])
    gesamt = endwert / startkapital - 1
    jahre = (kurve.index[-1] - kurve.index[0]).days / 365.25
    if jahre <= 0:
        jahre = len(kurve) / ppy
    # Annualisierung erst ab ca. 3 Monaten sinnvoll (sonst stark überzeichnete Werte)
    cagr = (endwert / startkapital) ** (1 / jahre) - 1 if jahre >= 0.24 and endwert > 0 else float("nan")
    mit_start = pd.concat([pd.Series([startkapital], index=[kurve.index[0] - pd.Timedelta(seconds=1)]), kurve])
    renditen = mit_start.pct_change().iloc[1:]
    max_dd = float((mit_start / mit_start.cummax() - 1).min())
    sharpe, _ = sharpe_ratio(renditen, ppy, rf_pa)
    vola = float(renditen.std(ddof=1) * math.sqrt(ppy)) if len(renditen) > 2 else float("nan")
    werte: dict[str, Any] = {
        "Startkapital": startkapital, "Endkapital": endwert, "Gesamtrendite": gesamt,
        "Annualisierte Rendite": cagr, "Maximaler Drawdown": max_dd, "Sharpe Ratio": sharpe,
        "Volatilität p. a.": vola, "Jahre": jahre,
    }
    if trades is not None:
        n = len(trades)
        gewinne = trades.loc[trades["Ergebnis"] > 0, "Ergebnis"] if n else pd.Series(dtype=float)
        verluste = trades.loc[trades["Ergebnis"] <= 0, "Ergebnis"] if n else pd.Series(dtype=float)
        brutto_verlust = float(-verluste.sum())
        if n == 0:
            profit_faktor = float("nan")
        elif brutto_verlust == 0:
            profit_faktor = float("inf") if gewinne.sum() > 0 else float("nan")
        else:
            profit_faktor = float(gewinne.sum()) / brutto_verlust
        serie = laengste = 0
        for ergebnis in trades["Ergebnis"] if n else []:
            serie = serie + 1 if ergebnis <= 0 else 0
            laengste = max(laengste, serie)
        werte.update({
            "Anzahl Trades": n,
            "Trefferquote": len(gewinne) / n if n else float("nan"),
            "Ø Gewinn": float(trades.loc[trades["Ergebnis"] > 0, "Rendite"].mean()) if len(gewinne) else float("nan"),
            "Ø Gewinn (Betrag)": float(gewinne.mean()) if len(gewinne) else float("nan"),
            "Ø Verlust": float(trades.loc[trades["Ergebnis"] <= 0, "Rendite"].mean()) if len(verluste) else float("nan"),
            "Ø Verlust (Betrag)": float(verluste.mean()) if len(verluste) else float("nan"),
            "Profit Factor": profit_faktor,
            "Längste Verlustserie": laengste,
            "Ø Haltedauer (Kerzen)": float(trades["Kerzen"].mean()) if n else float("nan"),
            "Bester Trade": float(trades["Rendite"].max()) if n else float("nan"),
            "Schlechtester Trade": float(trades["Rendite"].min()) if n else float("nan"),
            "Transaktionskosten": float(trades["Kosten"].sum()) if n else 0.0,
        })
    if position is not None and len(position):
        werte["Marktexposure"] = float(position.mean())
    return werte


def backtest_durchfuehren(d: pd.DataFrame, rp: RisikoParameter, bp: BacktestParameter | None = None,
                          intervall: IntervallInfo | None = None) -> BacktestErgebnis:
    """Simuliert die Signalstrategie (nur Long) und vergleicht sie mit Buy & Hold.

    Regeln gegen Look-Ahead-Bias:
    * Signale entstehen mit dem Schlusskurs der Kerze t und werden zum Eröffnungskurs von t+1 ausgeführt.
    * Stop-Loss (Einstieg − Multiplikator × ATR der Signalkerze) und Kursziel (Chance-Risiko-Verhältnis)
      werden ab der Einstiegskerze intrabar geprüft. Liegt die Eröffnung bereits jenseits der Marke (Gap),
      wird zum Eröffnungskurs ausgeführt. Werden Stop und Ziel in derselben Kerze erreicht, wird
      konservativ der Stop-Loss angenommen.
    * Transaktionskosten fallen bei jedem Kauf und Verkauf an, Slippage verschlechtert jeden Ausführungskurs.
    * Eine am Ende offene Position wird zum letzten Schlusskurs (inkl. Kosten) bewertet.
    """
    bp = bp or BacktestParameter()
    intervall = intervall or INTERVALLE["Täglich"]
    daten = d.dropna(subset=["Open", "High", "Low", "Close"])
    if len(daten) < 3:
        raise ValueError("Für einen Backtest werden mindestens drei Kerzen benötigt.")
    idx = daten.index
    eroeffnung, hoch, tief, schluss = (daten[s].to_numpy(dtype=float) for s in ("Open", "High", "Low", "Close"))
    atr = daten["ATR_14"].to_numpy(dtype=float) if "ATR_14" in daten else np.full(len(daten), np.nan)
    signal = daten["Signal"].to_numpy(dtype=int) if "Signal" in daten else np.zeros(len(daten), dtype=int)
    kosten = rp.transaktionskosten_pct / 100
    slip = rp.slippage_pct / 100
    n = len(daten)

    kasse = float(rp.kapital)
    stueck = 0
    position: dict[str, Any] = {}
    kapital_verlauf = np.zeros(n)
    im_markt = np.zeros(n, dtype=int)
    trades: list[dict[str, Any]] = []
    hinweise: list[str] = []
    kauf_order = verkauf_order = False
    zu_wenig_kapital = 0

    def schliessen(i: int, preis: float, grund: str) -> None:
        nonlocal kasse, stueck, position
        erloes = stueck * preis
        gebuehr = erloes * kosten
        kasse += erloes - gebuehr
        einsatz = position["einsatz"]
        ergebnis = erloes - gebuehr - einsatz
        trades.append({
            "Einstieg": position["datum"], "Einstiegskurs": position["preis"],
            "Ausstieg": idx[i], "Ausstiegskurs": preis, "Stück": stueck,
            "Stop-Loss": position["stop"], "Kursziel": position["ziel"],
            "Ergebnis": ergebnis, "Rendite": ergebnis / einsatz if einsatz else float("nan"),
            "Kosten": position["gebuehr"] + gebuehr, "Kerzen": i - position["i"],
            "Ausstiegsgrund": grund, "Signal": position["signal"],
        })
        stueck = 0
        position = {}

    for i in range(n):
        # 1) Orders aus der Vorkerze zum Eröffnungskurs ausführen
        if verkauf_order and stueck > 0:
            schliessen(i, eroeffnung[i] * (1 - slip), "Verkaufssignal")
        if kauf_order and stueck == 0:
            preis = eroeffnung[i] * (1 + slip)
            atr_signal = atr[i - 1] if i > 0 else np.nan
            risiko_abstand = rp.atr_multiplikator * atr_signal if math.isfinite(atr_signal) else np.nan
            stop = preis - risiko_abstand if bp.stop_loss_aktiv and math.isfinite(risiko_abstand) else np.nan
            ziel = (preis + rp.crv * risiko_abstand
                    if bp.kursziel_aktiv and math.isfinite(risiko_abstand) else np.nan)
            max_stueck = math.floor(kasse / (preis * (1 + kosten)))
            if bp.positionsgroesse == "risiko" and math.isfinite(risiko_abstand):
                menge, _ = positionsgroesse_berechnen(kasse, rp.risiko_pro_position_pct, preis,
                                                      preis - risiko_abstand, kosten, slip)
                menge = min(menge, max_stueck)
            else:
                menge = max_stueck
            if menge >= 1:
                gebuehr = menge * preis * kosten
                kasse -= menge * preis + gebuehr
                stueck = menge
                position = {"datum": idx[i], "preis": preis, "stop": stop, "ziel": ziel, "i": i,
                            "gebuehr": gebuehr, "einsatz": menge * preis + gebuehr,
                            "signal": SIGNAL_CODES.get(int(signal[i - 1]), "")}
            else:
                zu_wenig_kapital += 1
        kauf_order = verkauf_order = False

        # 2) Stop-Loss / Kursziel innerhalb der Kerze
        if stueck > 0:
            stop, ziel = position["stop"], position["ziel"]
            gap_moeglich = i > position["i"]
            if math.isfinite(stop) and tief[i] <= stop:
                basis = eroeffnung[i] if gap_moeglich and eroeffnung[i] < stop else stop
                schliessen(i, basis * (1 - slip), "Stop-Loss")
            elif math.isfinite(ziel) and hoch[i] >= ziel:
                basis = eroeffnung[i] if gap_moeglich and eroeffnung[i] > ziel else ziel
                schliessen(i, basis * (1 - slip), "Kursziel")

        # 3) Bewertung zum Schlusskurs
        kapital_verlauf[i] = kasse + stueck * schluss[i]
        im_markt[i] = int(stueck > 0)

        # 4) Signal am Kerzenschluss → Order für die nächste Kerze
        if i < n - 1:
            erlaubt = signal[i] == 2 if bp.nur_starke_signale else signal[i] > 0
            if erlaubt and stueck == 0:
                kauf_order = True
            elif signal[i] < 0 and stueck > 0:
                verkauf_order = True

    if stueck > 0:  # offene Position zum letzten Schlusskurs glattstellen
        schliessen(n - 1, schluss[-1] * (1 - slip), "Periodenende (offen)")
        kapital_verlauf[-1] = kasse
    if zu_wenig_kapital:
        hinweise.append(f"{zu_wenig_kapital} Kaufsignal(e) konnten mangels Kapital nicht umgesetzt werden.")

    # Buy & Hold: Kauf zum ersten Eröffnungskurs, Bewertung zum Schlusskurs, Glattstellung am Ende
    preis_bh = eroeffnung[0] * (1 + slip)
    stueck_bh = math.floor(rp.kapital / (preis_bh * (1 + kosten)))
    kasse_bh = rp.kapital - stueck_bh * preis_bh * (1 + kosten)
    kapital_bh = kasse_bh + stueck_bh * schluss
    kapital_bh[-1] = kasse_bh + stueck_bh * schluss[-1] * (1 - slip) * (1 - kosten)

    verlauf = pd.DataFrame({"Strategie": kapital_verlauf, "Buy & Hold": kapital_bh, "Position": im_markt}, index=idx)
    verlauf["Drawdown Strategie"] = verlauf["Strategie"] / verlauf["Strategie"].cummax().clip(lower=rp.kapital) - 1
    verlauf["Drawdown Buy & Hold"] = verlauf["Buy & Hold"] / verlauf["Buy & Hold"].cummax().clip(lower=rp.kapital) - 1
    trades_df = pd.DataFrame(trades, columns=["Einstieg", "Einstiegskurs", "Ausstieg", "Ausstiegskurs", "Stück",
                                              "Stop-Loss", "Kursziel", "Ergebnis", "Rendite", "Kosten", "Kerzen",
                                              "Ausstiegsgrund", "Signal"])
    ppy = perioden_pro_jahr_schaetzen(idx, intervall)
    rf = rp.risikofreier_zins_pct / 100
    werte_strategie = _kennzahlen_berechnen(verlauf["Strategie"], rp.kapital, ppy, rf, trades_df, verlauf["Position"])
    werte_bh = _kennzahlen_berechnen(verlauf["Buy & Hold"], rp.kapital, ppy, rf)
    werte_bh.update({"Anzahl Trades": 1, "Marktexposure": 1.0,
                     "Transaktionskosten": stueck_bh * preis_bh * kosten + stueck_bh * schluss[-1] * (1 - slip) * kosten})
    if len(trades_df) == 0:
        hinweise.append("Im Zeitraum wurde kein Trade ausgelöst – Parameter (z. B. Mindestbestätigungen) "
                        "lockern oder einen längeren Zeitraum wählen.")
    if werte_strategie["Jahre"] < 0.24:
        hinweise.append("Zeitraum unter drei Monaten: Die annualisierte Rendite wird nicht ausgewiesen.")
    elif werte_strategie["Jahre"] < 0.9:
        hinweise.append("Zeitraum unter einem Jahr: Annualisierte Werte sind nur eingeschränkt aussagekräftig.")
    return BacktestErgebnis(verlauf=verlauf, trades=trades_df,
                            kennzahlen=_kennzahlen_tabelle(werte_strategie, werte_bh),
                            werte={"Strategie": werte_strategie, "Buy & Hold": werte_bh}, hinweise=hinweise)


def _kennzahlen_tabelle(strategie: dict[str, Any], buy_hold: dict[str, Any]) -> pd.DataFrame:
    """Formatiert die Kennzahlen als Tabelle (Strategie vs. Buy & Hold)."""
    def betrag(x: Any) -> str:
        return fmt_zahl(x, 2)

    def prozent(x: Any) -> str:
        return fmt_pct(x, 2)

    def faktor(x: Any) -> str:
        return "∞" if isinstance(x, float) and math.isinf(x) else fmt_zahl(x, 2)

    zeilen = [
        ("Endkapital", betrag), ("Gesamtrendite", prozent), ("Annualisierte Rendite", prozent),
        ("Maximaler Drawdown", prozent), ("Sharpe Ratio", faktor), ("Volatilität p. a.", lambda x: fmt_pct(x, 2, False)),
        ("Anzahl Trades", lambda x: fmt_zahl(x, 0)), ("Trefferquote", lambda x: fmt_pct(x, 1, False)),
        ("Ø Gewinn", prozent), ("Ø Verlust", prozent), ("Profit Factor", faktor),
        ("Längste Verlustserie", lambda x: f"{int(x)} Trades" if _ist_zahl(x) else "–"),
        ("Ø Haltedauer (Kerzen)", lambda x: fmt_zahl(x, 1)), ("Marktexposure", lambda x: fmt_pct(x, 1, False)),
        ("Transaktionskosten", betrag),
    ]
    return pd.DataFrame([{"Kennzahl": name, "Strategie": f(strategie.get(name)), "Buy & Hold": f(buy_hold.get(name))}
                         for name, f in zeilen])


# =============================================================================
# 10) Analyse-Pipeline (verbindet die Einzelschritte)
# =============================================================================

@dataclass
class AnalyseErgebnis:
    """Alle Ergebnisse einer Analyse – Grundlage für Oberfläche, Bericht und Export."""

    d_voll: pd.DataFrame                 # alle Kerzen inkl. Vorlauf mit Indikatoren und Signalen
    d: pd.DataFrame                      # Kerzen im Anzeigezeitraum
    signale: pd.DataFrame                # Signale im Anzeigezeitraum
    trend: dict[str, Any]
    status: dict[str, Any]
    zonen: list[Zone]
    risiko: RisikoErgebnis
    backtest: BacktestErgebnis | None
    bewertung: tuple[str, str]
    punkte_tabelle: pd.DataFrame
    kauf_bedingungen: dict[str, bool]
    verkauf_bedingungen: dict[str, bool]
    letzte_kerze_vorlaeufig: bool
    volumen_verfuegbar: bool
    intervall: IntervallInfo
    sp: StrategieParameter
    rp: RisikoParameter
    bp: BacktestParameter
    hinweise: list[str] = field(default_factory=list)


def analyse_durchfuehren(daten: pd.DataFrame, anzeige_start: pd.Timestamp, intervall: IntervallInfo,
                         sp: StrategieParameter, rp: RisikoParameter, bp: BacktestParameter,
                         volumen_verfuegbar: bool = True, letzte_kerze_vorlaeufig: bool = False) -> AnalyseErgebnis:
    """Führt Indikatoren, Trend, Punkte, Signale, Risiko und Backtest nacheinander aus.

    Indikatoren werden auf allen Kerzen (inkl. Vorlauf) berechnet; Auswertung, Risiko und Backtest
    beziehen sich auf den Anzeigezeitraum.
    """
    hinweise: list[str] = []
    d_voll = indikatoren_berechnen(daten)
    d_voll, trend = trend_analyse(d_voll, sp)
    d_voll = signalpunkte_berechnen(d_voll, sp, volumen_verfuegbar)
    d_voll, signale = kauf_und_verkaufssignale_ermitteln(d_voll, sp, volumen_verfuegbar)

    d = d_voll.loc[d_voll.index >= anzeige_start]
    if len(d) < 2:
        d = d_voll.tail(max(2, min(len(d_voll), 30)))
    signale = signale.loc[signale["Datum"] >= d.index[0]].reset_index(drop=True)
    if len(d) < 30:
        hinweise.append(f"Nur {len(d)} Kerzen im Anzeigezeitraum – Kennzahlen sind wenig aussagekräftig.")
    if d["SMA_200"].isna().any():
        fehlend = int(d["SMA_200"].isna().sum())
        hinweise.append(f"SMA 200 ist für {fehlend} Kerze(n) des Zeitraums noch nicht berechenbar "
                        "(zu wenig Historie) – davon abhängige Bedingungen gelten dort als nicht erfüllt.")
    if letzte_kerze_vorlaeufig:
        hinweise.append("Die letzte Kerze ist noch nicht abgeschlossen (laufender Handel). Werte und Signale "
                        "dieser Kerze sind vorläufig und können sich bis zum Handelsschluss ändern.")

    zonen = unterstuetzung_widerstand_ermitteln(d)
    risiko = risiko_berechnen(d, rp, intervall, zonen)
    try:
        backtest = backtest_durchfuehren(d, rp, bp, intervall)
    except ValueError as exc:
        backtest = None
        hinweise.append(f"Backtest nicht möglich: {exc}")

    z = d_voll.iloc[-1]
    kauf_bed, verkauf_bed = signalbedingungen(d_voll, sp, volumen_verfuegbar)
    return AnalyseErgebnis(
        d_voll=d_voll, d=d, signale=signale, trend=trend, status=aktueller_signalstatus(d_voll, sp),
        zonen=zonen, risiko=risiko, backtest=backtest, bewertung=bewertung_einordnen(int(z["Punkte"]), sp),
        punkte_tabelle=punkte_aufschluesseln(z, sp, volumen_verfuegbar),
        kauf_bedingungen={k: bool(v.iloc[-1]) for k, v in kauf_bed.items()},
        verkauf_bedingungen={k: bool(v.iloc[-1]) for k, v in verkauf_bed.items()},
        letzte_kerze_vorlaeufig=letzte_kerze_vorlaeufig, volumen_verfuegbar=volumen_verfuegbar,
        intervall=intervall, sp=sp, rp=rp, bp=bp, hinweise=hinweise,
    )


# =============================================================================
# 11) Charts (Plotly)
# =============================================================================

def _x_werte(index: pd.DatetimeIndex, intraday: bool) -> Any:
    """Intraday: Kategorien-Achse ohne Nacht-/Wochenendlücken; sonst Datumsachse."""
    if intraday:
        return [ts.strftime("%d.%m.%y %H:%M") for ts in index]
    return index


def _hovertexte(d: pd.DataFrame, waehrung: str, intraday: bool) -> list[str]:
    """Tooltip je Kerze mit Datum, Kursen, Volumen und Indikatorwerten."""
    format_datum = "%d.%m.%Y %H:%M" if intraday else "%d.%m.%Y"

    def spalte(name: str) -> np.ndarray:
        return d[name].to_numpy(dtype=float) if name in d else np.full(len(d), np.nan)

    werte = {n: spalte(n) for n in ("Open", "High", "Low", "Close", "Volume", "Rel_Volumen", "SMA_20", "SMA_50",
                                     "SMA_200", "EMA_12", "EMA_26", "BB_Oben", "BB_Unten", "RSI_14", "Stoch_K",
                                     "Stoch_D", "MACD", "MACD_Signal", "ADX_14", "Plus_DI", "Minus_DI", "ATR_14")}
    vorschluss = np.r_[np.nan, werte["Close"][:-1]]
    punkte = d["Punkte"].to_numpy() if "Punkte" in d else np.zeros(len(d), dtype=int)
    bewertung = d["Bewertung"].to_numpy() if "Bewertung" in d else np.full(len(d), "")
    texte = []
    for i, ts in enumerate(d.index):
        w = {k: v[i] for k, v in werte.items()}
        veraenderung = w["Close"] / vorschluss[i] - 1 if vorschluss[i] else np.nan
        texte.append("<br>".join([
            f"<b>{ts.strftime(format_datum)}</b>",
            f"Eröffnung {fmt_zahl(w['Open'])} · Hoch {fmt_zahl(w['High'])} · Tief {fmt_zahl(w['Low'])}",
            f"Schluss <b>{fmt_zahl(w['Close'])} {waehrung}</b> ({fmt_pct(veraenderung)})",
            f"Volumen {fmt_volumen(w['Volume'])} ({fmt_zahl(w['Rel_Volumen'], 2)} × Ø)",
            f"SMA 20/50/200: {fmt_zahl(w['SMA_20'])} / {fmt_zahl(w['SMA_50'])} / {fmt_zahl(w['SMA_200'])}",
            f"EMA 12/26: {fmt_zahl(w['EMA_12'])} / {fmt_zahl(w['EMA_26'])}",
            f"Bollinger: {fmt_zahl(w['BB_Unten'])} – {fmt_zahl(w['BB_Oben'])}",
            f"RSI {fmt_zahl(w['RSI_14'], 1)} · Stoch %K/%D {fmt_zahl(w['Stoch_K'], 1)}/{fmt_zahl(w['Stoch_D'], 1)}",
            f"MACD {fmt_zahl(w['MACD'], 3)} · Signal {fmt_zahl(w['MACD_Signal'], 3)}",
            f"ADX {fmt_zahl(w['ADX_14'], 1)} (+DI {fmt_zahl(w['Plus_DI'], 1)} / −DI {fmt_zahl(w['Minus_DI'], 1)})"
            f" · ATR {fmt_zahl(w['ATR_14'])}",
            f"Punkte {fmt_punkte(punkte[i])} ({bewertung[i]})",
        ]))
    return texte


def _feiertage(index: pd.DatetimeIndex) -> list[str]:
    """Werktage ohne Kerze (Feiertage, Handelsausfälle) für Plotly-rangebreaks."""
    if len(index) < 2:
        return []
    werktage = pd.bdate_range(index.min(), index.max())
    return [t.strftime("%Y-%m-%d") for t in werktage.difference(index.normalize())]


def chart_erstellen(d: pd.DataFrame, signale: pd.DataFrame | None = None, zonen: Sequence[Zone] | None = None,
                    overlays: Sequence[str] = tuple(OVERLAY_STANDARD), panels: Sequence[str] = tuple(PANEL_STANDARD),
                    sp: StrategieParameter | None = None, titel: str = "", intervall: IntervallInfo | None = None,
                    waehrung: str = "TRY") -> go.Figure:
    """Interaktiver Candlestick-Chart mit Overlays, Signalmarkierungen und Indikator-Fenstern.

    Einzelne Elemente lassen sich über die Auswahl in der Seitenleiste oder per Klick auf die Legende
    ein- und ausblenden.
    """
    sp = sp or StrategieParameter()
    intervall = intervall or INTERVALLE["Täglich"]
    intraday = intervall.intraday
    panels = [p for p in PANEL_OPTIONEN if p in panels]
    zeilen = 1 + len(panels)
    hoehen = [0.56] + [0.44 / len(panels)] * len(panels) if panels else [1.0]
    fig = make_subplots(rows=zeilen, cols=1, shared_xaxes=True, vertical_spacing=0.025, row_heights=hoehen)
    x = _x_werte(d.index, intraday)

    fig.add_trace(go.Candlestick(
        x=x, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name="Kurs",
        increasing=dict(line=dict(color=FARBEN["kerze_hoch"], width=1), fillcolor=FARBEN["kerze_hoch"]),
        decreasing=dict(line=dict(color=FARBEN["kerze_tief"], width=1), fillcolor=FARBEN["kerze_tief"]),
        hovertext=_hovertexte(d, waehrung, intraday), hoverinfo="text",
    ), row=1, col=1)

    linien = [("SMA 20", "SMA_20", FARBEN["sma20"], 1.3, "solid"), ("SMA 50", "SMA_50", FARBEN["sma50"], 1.5, "solid"),
              ("SMA 200", "SMA_200", FARBEN["sma200"], 1.8, "solid"), ("EMA 12", "EMA_12", FARBEN["ema12"], 1.1, "dot"),
              ("EMA 26", "EMA_26", FARBEN["ema26"], 1.1, "dot")]
    for name, spalte, farbe, breite, strich in linien:
        if name in overlays and spalte in d:
            fig.add_trace(go.Scatter(x=x, y=d[spalte], name=name, mode="lines", hoverinfo="skip",
                                     line=dict(color=farbe, width=breite, dash=strich)), row=1, col=1)
    if "Bollinger-Bänder" in overlays and "BB_Oben" in d:
        fig.add_trace(go.Scatter(x=x, y=d["BB_Oben"], name="Bollinger-Bänder", mode="lines", hoverinfo="skip",
                                 legendgroup="bb", showlegend=False,
                                 line=dict(color=FARBEN["bb"], width=1, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=d["BB_Unten"], name="Bollinger-Bänder", mode="lines", hoverinfo="skip",
                                 legendgroup="bb", fill="tonexty", fillcolor="rgba(148,163,184,0.10)",
                                 line=dict(color=FARBEN["bb"], width=1, dash="dash")), row=1, col=1)

    if "Unterstützung/Widerstand" in overlays:
        spanne = float(d["High"].max() - d["Low"].min()) or 1.0
        letzte_beschriftung: float | None = None
        for zone in sorted(zonen or [], key=lambda z: z.mitte):
            rgb = "22,163,74" if zone.art == "Unterstützung" else "220,38,38"
            kurz = "U" if zone.art == "Unterstützung" else "W"
            beschriften = letzte_beschriftung is None or abs(zone.mitte - letzte_beschriftung) > 0.035 * spanne
            beschriftung: dict[str, Any] = {}
            if beschriften:   # nahe beieinanderliegende Zonen nur einmal beschriften
                beschriftung = dict(annotation_text=f"{kurz} {fmt_zahl(zone.mitte)} ({zone.beruehrungen}×)",
                                    annotation_position="top left", annotation_font_size=10,
                                    annotation_font_color=f"rgb({rgb})")
                letzte_beschriftung = zone.mitte
            fig.add_hrect(y0=zone.unten, y1=zone.oben, row=1, col=1, line_width=0, layer="below",
                          fillcolor=f"rgba({rgb},0.14)", **beschriftung)

    if "Signale" in overlays and signale is not None and not signale.empty:
        stile = [(KAUF_STARK, "star-triangle-up", 17, STIL_FARBEN["stark_pos"]),
                 (KAUF, "triangle-up", 12, FARBEN["positiv"]),
                 (VERKAUF, "triangle-down", 12, FARBEN["negativ"]),
                 (VERKAUF_STARK, "star-triangle-down", 17, STIL_FARBEN["stark_neg"])]
        ersatz_abstand = float((d["High"] - d["Low"]).mean())
        for typ, symbol, groesse, farbe in stile:
            teil = signale[signale["Signaltyp"] == typ]
            teil = teil[teil["Datum"].isin(d.index)]
            if teil.empty:
                continue
            ist_kauf = typ in (KAUF_STARK, KAUF)
            daten_teil = d.loc[pd.DatetimeIndex(teil["Datum"])]
            abstand = daten_teil["ATR_14"].fillna(ersatz_abstand).to_numpy() * 0.7
            y = daten_teil["Low"].to_numpy() - abstand if ist_kauf else daten_teil["High"].to_numpy() + abstand
            hover = [f"<b>{r['Signaltyp']}</b><br>{fmt_datum(r['Datum'], intraday)} · Kurs {fmt_zahl(r['Kurs'])}"
                     f"<br>Punktzahl {fmt_punkte(r['Punktzahl'])} · Stärke {r['Signalstärke']}"
                     f"<br>{_umbrechen(r['Erklärung'], 70)}" for _, r in teil.iterrows()]
            fig.add_trace(go.Scatter(x=_x_werte(daten_teil.index, intraday), y=y, mode="markers", name=typ,
                                     hovertext=hover, hoverinfo="text",
                                     marker=dict(symbol=symbol, size=groesse, color=farbe,
                                                 line=dict(width=1, color="rgba(0,0,0,0.35)"))), row=1, col=1)

    for nummer, panel in enumerate(panels, start=2):
        _panel_zeichnen(fig, d, x, panel, nummer, sp)

    fig.update_layout(
        height=520 + 160 * len(panels), hovermode="x unified", separators=",.", dragmode="zoom",
        margin=dict(l=10, r=10, t=110 if titel else 40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.005, xanchor="left", x=0, font=dict(size=11)),
        title=dict(text=titel, x=0, y=0.995, yanchor="top", font=dict(size=15)) if titel else None,
        hoverlabel=dict(namelength=-1, font_size=12), bargap=0.1,
    )
    fig.update_xaxes(rangeslider_visible=False, showspikes=True, spikemode="across", spikesnap="cursor",
                     spikethickness=1, spikedash="dot", spikecolor="#94a3b8", showgrid=True)
    fig.update_yaxes(title_text=f"Kurs ({waehrung})", row=1, col=1)
    if intraday:
        fig.update_xaxes(type="category", nticks=10, tickangle=0)
    elif intervall.code == "1d":
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(values=_feiertage(d.index))],
                         hoverformat="%d.%m.%Y")
    else:
        fig.update_xaxes(hoverformat="%d.%m.%Y")
    return fig


def _umbrechen(text: str, breite: int = 70) -> str:
    """Bricht lange Tooltip-Texte in mehrere Zeilen um."""
    woerter, zeilen, aktuell = str(text).split(), [], ""
    for wort in woerter:
        if len(aktuell) + len(wort) + 1 > breite and aktuell:
            zeilen.append(aktuell)
            aktuell = wort
        else:
            aktuell = f"{aktuell} {wort}".strip()
    if aktuell:
        zeilen.append(aktuell)
    return "<br>".join(zeilen)


def _panel_zeichnen(fig: go.Figure, d: pd.DataFrame, x: Any, panel: str, zeile: int, sp: StrategieParameter) -> None:
    """Zeichnet ein Indikator-Fenster unterhalb des Kurscharts."""
    def linie(spalte: str, name: str, farbe: str, breite: float = 1.4, strich: str = "solid", stellen: int = 2) -> None:
        fig.add_trace(go.Scatter(x=x, y=d[spalte], name=name, mode="lines", showlegend=False,
                                 line=dict(color=farbe, width=breite, dash=strich),
                                 hovertemplate=f"{name}: %{{y:,.{stellen}f}}<extra></extra>"), row=zeile, col=1)

    def waagerecht(y: float, farbe: str = "#94a3b8", strich: str = "dash") -> None:
        fig.add_hline(y=y, row=zeile, col=1, line_width=1, line_dash=strich, line_color=farbe)

    if panel == "Volumen":
        farben = np.where(d["Close"] >= d["Open"], "rgba(22,163,74,0.55)", "rgba(220,38,38,0.55)")
        fig.add_trace(go.Bar(x=x, y=d["Volume"], name="Volumen", marker_color=farben, showlegend=False,
                             hovertemplate="Volumen: %{y:,.0f}<extra></extra>"), row=zeile, col=1)
        linie("Volumen_SMA_20", "Ø Volumen (20)", FARBEN["linie2"], 1.2, stellen=0)
        fig.update_yaxes(title_text="Volumen", row=zeile, col=1)
    elif panel == "RSI":
        linie("RSI_14", "RSI 14", "#8b5cf6", 1.5, stellen=1)
        waagerecht(sp.rsi_ueberkauft, FARBEN["negativ"])
        waagerecht(sp.rsi_ueberverkauft, FARBEN["positiv"])
        waagerecht(50, strich="dot")
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=zeile, col=1)
    elif panel == "MACD":
        farben = np.where(d["MACD_Hist"] >= 0, "rgba(22,163,74,0.6)", "rgba(220,38,38,0.6)")
        fig.add_trace(go.Bar(x=x, y=d["MACD_Hist"], name="Histogramm", marker_color=farben, showlegend=False,
                             hovertemplate="Histogramm: %{y:,.3f}<extra></extra>"), row=zeile, col=1)
        linie("MACD", "MACD", FARBEN["linie1"], 1.4, stellen=3)
        linie("MACD_Signal", "Signallinie", FARBEN["linie2"], 1.2, stellen=3)
        fig.update_yaxes(title_text="MACD", row=zeile, col=1)
    elif panel == "Stochastik":
        linie("Stoch_K", "%K", FARBEN["linie1"], 1.4, stellen=1)
        linie("Stoch_D", "%D", FARBEN["linie2"], 1.2, stellen=1)
        waagerecht(80, FARBEN["negativ"])
        waagerecht(20, FARBEN["positiv"])
        fig.update_yaxes(title_text="Stoch.", range=[0, 100], row=zeile, col=1)
    elif panel == "ADX":
        linie("ADX_14", "ADX", "#64748b", 1.8, stellen=1)
        linie("Plus_DI", "+DI", FARBEN["positiv"], 1.1, stellen=1)
        linie("Minus_DI", "−DI", FARBEN["negativ"], 1.1, stellen=1)
        waagerecht(sp.adx_schwelle)
        fig.update_yaxes(title_text="ADX", row=zeile, col=1)
    elif panel == "OBV":
        linie("OBV", "OBV", FARBEN["linie1"], 1.4, stellen=0)
        linie("OBV_SMA_20", "OBV Ø 20", FARBEN["linie2"], 1.0, "dot", stellen=0)
        fig.update_yaxes(title_text="OBV", row=zeile, col=1)
    elif panel == "ATR":
        linie("ATR_14", "ATR 14", "#0ea5e9", 1.4)
        fig.update_yaxes(title_text="ATR", row=zeile, col=1)


def backtest_chart_erstellen(bt: BacktestErgebnis, waehrung: str = "TRY", intraday: bool = False) -> go.Figure:
    """Kapitalentwicklung (Strategie vs. Buy & Hold) mit Ein-/Ausstiegen und Drawdown."""
    v = bt.verlauf
    x = _x_werte(v.index, intraday)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    fig.add_trace(go.Scatter(x=x, y=v["Strategie"], name="Signalstrategie", line=dict(color=FARBEN["linie1"], width=2),
                             hovertemplate="Strategie: %{y:,.0f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=v["Buy & Hold"], name="Buy & Hold", line=dict(color=FARBEN["grau"], width=1.6),
                             hovertemplate="Buy & Hold: %{y:,.0f}<extra></extra>"), row=1, col=1)
    if not bt.trades.empty:
        for spalte, name, symbol, farbe in (("Einstieg", "Einstieg", "triangle-up", FARBEN["positiv"]),
                                            ("Ausstieg", "Ausstieg", "triangle-down", FARBEN["negativ"])):
            zeitpunkte = pd.DatetimeIndex(bt.trades[spalte])
            fig.add_trace(go.Scatter(x=_x_werte(zeitpunkte, intraday), y=v["Strategie"].reindex(zeitpunkte),
                                     mode="markers", name=name, marker=dict(symbol=symbol, size=9, color=farbe),
                                     hovertemplate=f"{name}: %{{y:,.0f}}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=v["Drawdown Strategie"] * 100, name="Drawdown Strategie", fill="tozeroy",
                             line=dict(color=FARBEN["negativ"], width=1), fillcolor="rgba(220,38,38,0.18)",
                             hovertemplate="DD Strategie: %{y:,.1f} %<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=v["Drawdown Buy & Hold"] * 100, name="Drawdown Buy & Hold",
                             line=dict(color=FARBEN["grau"], width=1, dash="dot"),
                             hovertemplate="DD Buy & Hold: %{y:,.1f} %<extra></extra>"), row=2, col=1)
    fig.update_layout(height=520, hovermode="x unified", separators=",.", margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0))
    fig.update_yaxes(title_text=f"Kapital ({waehrung})", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    if intraday:
        fig.update_xaxes(type="category", nticks=10)
    return fig


def punkte_anzeige_erstellen(punkte: int, sp: StrategieParameter) -> go.Figure:
    """Tachometer der Gesamtpunktzahl mit farbigen Bewertungsbereichen."""
    g = sp.gewichte
    unten, oben = min(g.minimum(), sp.schwelle_stark_negativ - 1), max(g.maximum(), sp.schwelle_stark_positiv + 1)
    farbe = STIL_FARBEN[bewertung_einordnen(punkte, sp)[1]]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=punkte,
        number=dict(valueformat="+d" if punkte else "d", font=dict(size=40, color=farbe)),
        gauge=dict(
            axis=dict(range=[unten, oben], tickmode="linear", dtick=3),
            bar=dict(color=farbe, thickness=0.3), borderwidth=0,
            steps=[dict(range=[unten, sp.schwelle_stark_negativ], color="rgba(185,28,28,0.35)"),
                   dict(range=[sp.schwelle_stark_negativ, sp.schwelle_negativ], color="rgba(220,38,38,0.18)"),
                   dict(range=[sp.schwelle_negativ, sp.schwelle_positiv], color="rgba(217,119,6,0.18)"),
                   dict(range=[sp.schwelle_positiv, sp.schwelle_stark_positiv], color="rgba(22,163,74,0.18)"),
                   dict(range=[sp.schwelle_stark_positiv, oben], color="rgba(21,128,61,0.35)")],
        ),
    ))
    fig.update_layout(height=200, margin=dict(l=25, r=25, t=15, b=5))
    return fig


# =============================================================================
# 12) Bericht (verständliche Zusammenfassung)
# =============================================================================

def _rsi_einordnung(rsi: float, sp: StrategieParameter) -> str:
    if not _ist_zahl(rsi):
        return "nicht verfügbar"
    if rsi < sp.rsi_ueberverkauft:
        return "überverkauft"
    if rsi > sp.rsi_ueberkauft:
        return "überkauft"
    return "neutral"


def kurzfazit_erstellen(erg: AnalyseErgebnis, name: str, symbol: str, einheit: str,
                        uebersicht: dict[str, Any]) -> str:
    """Kurze, verständliche Zusammenfassung der aktuellen technischen Lage."""
    z = erg.d_voll.iloc[-1]
    h = erg.trend["horizonte"]
    teile = [f"{name} ({symbol}) notiert bei {fmt_zahl(uebersicht['kurs'])} {einheit} "
             f"({fmt_pct(uebersicht['veraenderung_pct'])} zum Vortag)."]
    teile.append(f"Der mittelfristige Trend ist {h['mittel']['text']}, der langfristige {h['lang']['text']}; "
                 f"Trendstärke: {erg.trend['staerke']}.")
    macd_lage = "über" if z["MACD"] > z["MACD_Signal"] else "unter"
    teile.append(f"Der RSI liegt bei {fmt_zahl(z['RSI_14'], 0)} ({_rsi_einordnung(z['RSI_14'], erg.sp)}), "
                 f"der MACD {macd_lage} seiner Signallinie.")
    teile.append(f"Technische Gesamtbewertung: {erg.bewertung[0]} ({punkte_text(z['Punkte'])}).")
    status = erg.status
    if status["code"] != 0:
        alter = "auf der letzten Kerze" if status["alter"] == 0 else f"vor {status['alter']} Kerze(n)"
        teile.append(f"Aktuell liegt ein {status['status']} vor ({alter}, "
                     f"{fmt_datum(status['datum'], erg.intervall.intraday)}).")
    else:
        teile.append("Aktuell gibt es kein frisches Kauf- oder Verkaufssignal – Halten/Beobachten.")
    return " ".join(teile)


def bericht_erstellen(erg: AnalyseErgebnis, name: str, symbol: str, quelle: str, zeitraum: str,
                      intervall_name: str, waehrung: str, uebersicht: dict[str, Any],
                      qualitaet: DatenQualitaet | None = None, einheit: str | None = None) -> str:
    """Erstellt einen Markdown-Bericht mit allen wichtigen Ergebnissen (Kurse in ``einheit``)."""
    einheit = einheit or waehrung
    z = erg.d_voll.iloc[-1]
    r = erg.risiko
    intraday = erg.intervall.intraday
    zeilen = [
        f"# Technische Analyse: {name} ({symbol})",
        "",
        f"*Erstellt am {datetime.now().strftime('%d.%m.%Y %H:%M')} · Datenstand: "
        f"{fmt_datum(erg.d.index[-1], intraday)} · Intervall: {intervall_name} · Zeitraum: {zeitraum} · "
        f"Quelle: {quelle} · Währung: {waehrung}*",
        "",
        f"> {HAFTUNGSAUSSCHLUSS}",
        "",
        "## Kurzfazit",
        "",
        kurzfazit_erstellen(erg, name, symbol, einheit, uebersicht),
        "",
        "## Kurs",
        "",
        f"- Letzter Kurs: **{fmt_zahl(uebersicht['kurs'])} {einheit}** ({fmt_pct(uebersicht['veraenderung_pct'])})",
        f"- Volumen (letzter Handelstag): {fmt_volumen(uebersicht['volumen'])} Stück, "
        f"Umsatz ca. {fmt_volumen(uebersicht['umsatz'])} {waehrung}",
        f"- 52-Wochen-Spanne: {fmt_spanne(uebersicht['tief_52w'], uebersicht['hoch_52w'])} {einheit}",
        "",
        "## Trend",
        "",
    ]
    for schluessel, titel in (("kurz", "Kurzfristig"), ("mittel", "Mittelfristig"), ("lang", "Langfristig")):
        horizont = erg.trend["horizonte"][schluessel]
        zeilen.append(f"- {titel}: **{horizont['text']}** ({horizont['grundlage']})")
    zeilen += [f"- Trendstärke: {erg.trend['staerke']} – {erg.trend['richtung']}", "", "## Indikatoren", ""]
    zeilen += [f"- {eintrag['Indikator']}: {eintrag['Wert']} – {eintrag['Einordnung']}"
               for eintrag in indikator_uebersicht(erg.d_voll, erg.sp, erg.volumen_verfuegbar)]
    zeilen += ["", f"## Punktbewertung: {punkte_text(z['Punkte'])} → {erg.bewertung[0]}", "",
               "| Komponente | Wert | Punkte | Begründung |", "|---|---|---:|---|"]
    zeilen += [f"| {p['Komponente']} | {p['Wert']} | {fmt_punkte(p['Punkte'])} | {p['Begründung']} |"
               for _, p in erg.punkte_tabelle.iterrows()]
    zeilen += ["", f"## Signalstatus: {erg.status['status']}", ""]
    for titel, bedingungen in (("Kaufbedingungen", erg.kauf_bedingungen), ("Verkaufsbedingungen", erg.verkauf_bedingungen)):
        erfuellt = sum(bedingungen.values())
        zeilen.append(f"- {titel} ({erfuellt}/{len(bedingungen)} erfüllt): "
                      + ", ".join(f"{'✓' if ok else '✗'} {n}" for n, ok in bedingungen.items()))
    if not erg.signale.empty:
        zeilen += ["", "Letzte Signale im Zeitraum:", ""]
        for _, s in erg.signale.tail(5).iloc[::-1].iterrows():
            zeilen.append(f"- {fmt_datum(s['Datum'], intraday)}: **{s['Signaltyp']}** bei {fmt_zahl(s['Kurs'])} "
                          f"({punkte_text(s['Punktzahl'])}, Stärke {s['Signalstärke']})")
    rp = erg.rp
    zeilen += [
        "", "## Risikoanalyse (Long-Position)", "",
        f"- ATR 14: {fmt_zahl(r.atr)} {einheit} ({fmt_pct(r.atr_prozent, 2, False)} des Kurses)",
        f"- Stop-Loss ({fmt_zahl(rp.atr_multiplikator, 1)} × ATR): {fmt_zahl(r.stop_loss)} {einheit} "
        f"({fmt_pct(r.stop_abstand)})",
        f"- Kursziel (CRV {fmt_zahl(rp.crv, 1)}): {fmt_zahl(r.kursziel)} {einheit} ({fmt_pct(r.ziel_abstand)})",
        f"- Positionsgröße bei {fmt_zahl(rp.kapital, 0)} {waehrung} Kapital und {fmt_zahl(rp.risiko_pro_position_pct, 1)} % "
        f"Risiko: {r.stueckzahl} Stück ≈ {fmt_zahl(r.positionswert, 0)} {waehrung} "
        f"({fmt_pct(r.positionsanteil, 1, False)} des Kapitals), max. Verlust ≈ {fmt_zahl(r.max_verlust, 0)} {waehrung}",
        f"- Historische Volatilität p. a.: {fmt_pct(r.volatilitaet_pa, 1, False)} (letzte 20 Kerzen: "
        f"{fmt_pct(r.volatilitaet_20_pa, 1, False)})",
        f"- Historischer VaR (95 %, 1 Kerze): {fmt_pct(r.var_95, 2, False)}",
        f"- Längste Verlustserie: {r.verlustserie_laenge} Kerzen in Folge ({fmt_pct(r.verlustserie_verlust)}, "
        f"{fmt_datum(r.verlustserie_start, intraday)} – {fmt_datum(r.verlustserie_ende, intraday)})",
        f"- Maximaler Drawdown: {fmt_pct(r.max_drawdown)} (Hoch {fmt_datum(r.dd_hoch, intraday)}, Tief "
        f"{fmt_datum(r.dd_tief, intraday)}, "
        + (f"erholt am {fmt_datum(r.dd_erholung, intraday)})" if r.dd_erholung is not None else "noch nicht erholt)"),
        f"- Sharpe Ratio (Buy & Hold): {fmt_zahl(r.sharpe, 2) if r.sharpe is not None else r.sharpe_hinweis}",
    ]
    zeilen += [f"- Hinweis: {h}" for h in r.hinweise]
    if erg.backtest is not None:
        zeilen += ["", "## Backtest (Signalstrategie vs. Buy & Hold)", "",
                   "| Kennzahl | Strategie | Buy & Hold |", "|---|---:|---:|"]
        zeilen += [f"| {k['Kennzahl']} | {k['Strategie']} | {k['Buy & Hold']} |"
                   for _, k in erg.backtest.kennzahlen.iterrows()]
        zeilen += ["", f"*{BACKTEST_HINWEIS}*"]
    hinweise = list(erg.hinweise) + (qualitaet.warnungen if qualitaet else [])
    zeilen += ["", "## Hinweise", ""] + [f"- {h}" for h in hinweise] + [f"- {DATENHINWEIS}", f"- {HAFTUNGSAUSSCHLUSS}"]
    return "\n".join(zeilen)


def indikator_uebersicht(d: pd.DataFrame, sp: StrategieParameter, volumen_verfuegbar: bool = True) -> list[dict[str, str]]:
    """Aktuelle Indikatorwerte mit Einordnung und Farbstil (pos/neg/neutral)."""
    z = d.iloc[-1]
    kurs = z["Close"]
    eintraege: list[dict[str, str]] = []

    def neu(indikator: str, wert: str, einordnung: str, stil: str) -> None:
        eintraege.append({"Indikator": indikator, "Wert": wert, "Einordnung": einordnung, "Stil": stil})

    rsi = z["RSI_14"]
    if _ist_zahl(rsi):
        if rsi < sp.rsi_ueberverkauft:
            neu("RSI 14", fmt_zahl(rsi, 1), "überverkauft – Erholungspotenzial", "pos")
        elif rsi > sp.rsi_ueberkauft:
            neu("RSI 14", fmt_zahl(rsi, 1), "überkauft – Rücksetzergefahr", "neg")
        else:
            neu("RSI 14", fmt_zahl(rsi, 1), "neutral", "neutral")
    if _ist_zahl(z["MACD"]) and _ist_zahl(z["MACD_Signal"]):
        bullisch = z["MACD"] > z["MACD_Signal"]
        neu("MACD (12/26/9)", f"{fmt_zahl(z['MACD'], 3)} / {fmt_zahl(z['MACD_Signal'], 3)}",
            "über Signallinie (bullisch)" if bullisch else "unter Signallinie (bärisch)", "pos" if bullisch else "neg")
    if _ist_zahl(z["Stoch_K"]):
        k = z["Stoch_K"]
        stil, text = ("neg", "überkauft (> 80)") if k > 80 else ("pos", "überverkauft (< 20)") if k < 20 \
            else ("neutral", "neutral")
        neu("Stochastik %K/%D", f"{fmt_zahl(k, 1)} / {fmt_zahl(z['Stoch_D'], 1)}", text, stil)
    if _ist_zahl(z["ADX_14"]):
        adx = z["ADX_14"]
        if adx >= sp.adx_schwelle:
            aufwaerts = z["Plus_DI"] > z["Minus_DI"]
            neu("ADX 14", fmt_zahl(adx, 1), f"starker Trend ({'aufwärts' if aufwaerts else 'abwärts'})",
                "pos" if aufwaerts else "neg")
        else:
            neu("ADX 14", fmt_zahl(adx, 1), "kein ausgeprägter Trend", "neutral")
    if _ist_zahl(z["ATR_14"]):
        neu("ATR 14", fmt_zahl(z["ATR_14"]), f"{fmt_pct(z['ATR_14'] / kurs, 2, False)} des Kurses", "info")
    for name, spalte in (("SMA 20", "SMA_20"), ("SMA 50", "SMA_50"), ("SMA 200", "SMA_200")):
        if _ist_zahl(z[spalte]):
            ueber = kurs > z[spalte]
            neu(name, fmt_zahl(z[spalte]), f"Kurs {'über' if ueber else 'unter'} {name} ({fmt_pct(kurs / z[spalte] - 1)})",
                "pos" if ueber else "neg")
        else:
            neu(name, "–", "noch nicht berechenbar", "neutral")
    if _ist_zahl(z["EMA_12"]) and _ist_zahl(z["EMA_26"]):
        ueber = z["EMA_12"] > z["EMA_26"]
        neu("EMA 12 / EMA 26", f"{fmt_zahl(z['EMA_12'])} / {fmt_zahl(z['EMA_26'])}",
            "EMA 12 über EMA 26" if ueber else "EMA 12 unter EMA 26", "pos" if ueber else "neg")
    if _ist_zahl(z["BB_ProzentB"]):
        pb = z["BB_ProzentB"]
        stil, text = ("neg", "über dem oberen Band (überdehnt)") if pb > 1 else \
            ("pos", "unter dem unteren Band (überdehnt)") if pb < 0 else ("neutral", "innerhalb der Bänder")
        neu("Bollinger %B", fmt_zahl(pb, 2), f"{text}, Bandbreite {fmt_zahl(z['BB_Bandbreite'], 1)} %", stil)
    if volumen_verfuegbar:
        if _ist_zahl(z["Rel_Volumen"]):
            rv = z["Rel_Volumen"]
            text = "erhöht" if rv >= sp.volumen_faktor else "niedrig" if rv < 0.7 else "normal"
            neu("Relative Volumenstärke", f"{fmt_zahl(rv, 2)} × Ø 20", text, "info")
        if _ist_zahl(z["OBV_SMA_20"]):
            steigend = z["OBV"] > z["OBV_SMA_20"]
            neu("OBV", fmt_volumen(z["OBV"]), "über Ø 20 – Kaufdruck" if steigend else "unter Ø 20 – Verkaufsdruck",
                "pos" if steigend else "neg")
    return eintraege


# =============================================================================
# 13) SQLite-Speicherung (optional) und Export
# =============================================================================

DB_DATEI = Path(__file__).resolve().with_name("bist_analysen.sqlite")

_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS analysen (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    zeitpunkt TEXT NOT NULL,
    symbol TEXT NOT NULL,
    name TEXT,
    quelle TEXT,
    zeitraum TEXT,
    intervall TEXT,
    waehrung TEXT,
    datenstand TEXT,
    kurs REAL,
    punkte INTEGER,
    bewertung TEXT,
    signalstatus TEXT,
    stop_loss REAL,
    kursziel REAL,
    rendite_strategie REAL,
    rendite_buy_hold REAL,
    bericht TEXT
);
CREATE TABLE IF NOT EXISTS signale (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analyse_id INTEGER NOT NULL REFERENCES analysen(id) ON DELETE CASCADE,
    datum TEXT,
    kurs REAL,
    signaltyp TEXT,
    punktzahl INTEGER,
    indikatoren TEXT,
    staerke TEXT,
    erklaerung TEXT
);
"""


def datenbank_oeffnen(pfad: Path | str = DB_DATEI) -> sqlite3.Connection:
    """Öffnet (und erstellt bei Bedarf) die SQLite-Datenbank."""
    verbindung = sqlite3.connect(str(pfad))
    verbindung.execute("PRAGMA foreign_keys = ON")
    verbindung.executescript(_DB_SCHEMA)
    return verbindung


def _wert_oder_none(x: Any) -> float | None:
    return float(x) if _ist_zahl(x) else None


def analyse_speichern(erg: AnalyseErgebnis, name: str, symbol: str, quelle: str, zeitraum: str,
                      intervall_name: str, waehrung: str, bericht: str, pfad: Path | str = DB_DATEI) -> int:
    """Speichert Zusammenfassung und Signale einer Analyse; liefert die ID des Eintrags."""
    z = erg.d_voll.iloc[-1]
    bt = erg.backtest.werte if erg.backtest is not None else {"Strategie": {}, "Buy & Hold": {}}
    with closing(datenbank_oeffnen(pfad)) as db, db:
        cursor = db.execute(
            "INSERT INTO analysen (zeitpunkt, symbol, name, quelle, zeitraum, intervall, waehrung, datenstand, kurs, "
            "punkte, bewertung, signalstatus, stop_loss, kursziel, rendite_strategie, rendite_buy_hold, bericht) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), symbol, name, quelle, zeitraum, intervall_name, waehrung,
             pd.Timestamp(erg.d.index[-1]).isoformat(), float(z["Close"]), int(z["Punkte"]), erg.bewertung[0],
             erg.status["status"], _wert_oder_none(erg.risiko.stop_loss), _wert_oder_none(erg.risiko.kursziel),
             _wert_oder_none(bt["Strategie"].get("Gesamtrendite")), _wert_oder_none(bt["Buy & Hold"].get("Gesamtrendite")),
             bericht))
        analyse_id = int(cursor.lastrowid)
        db.executemany(
            "INSERT INTO signale (analyse_id, datum, kurs, signaltyp, punktzahl, indikatoren, staerke, erklaerung) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(analyse_id, pd.Timestamp(s["Datum"]).isoformat(), float(s["Kurs"]), s["Signaltyp"], int(s["Punktzahl"]),
              s["Auslösende Indikatoren"], s["Signalstärke"], s["Erklärung"]) for _, s in erg.signale.iterrows()])
    return analyse_id


def analysen_laden(pfad: Path | str = DB_DATEI, limit: int = 200) -> pd.DataFrame:
    """Liest die zuletzt gespeicherten Analysen."""
    if not Path(pfad).exists():
        return pd.DataFrame()
    with closing(datenbank_oeffnen(pfad)) as db:
        return pd.read_sql_query(
            "SELECT id, zeitpunkt, symbol, name, quelle, zeitraum, intervall, waehrung, datenstand, kurs, punkte, "
            "bewertung, signalstatus, stop_loss, kursziel, rendite_strategie, rendite_buy_hold "
            "FROM analysen ORDER BY id DESC LIMIT ?", db, params=(int(limit),))


def gespeicherte_signale_laden(analyse_id: int, pfad: Path | str = DB_DATEI) -> pd.DataFrame:
    with closing(datenbank_oeffnen(pfad)) as db:
        return pd.read_sql_query("SELECT datum, kurs, signaltyp, punktzahl, indikatoren, staerke, erklaerung "
                                 "FROM signale WHERE analyse_id = ? ORDER BY datum", db, params=(int(analyse_id),))


def analysen_loeschen(pfad: Path | str = DB_DATEI) -> None:
    with closing(datenbank_oeffnen(pfad)) as db, db:
        db.execute("DELETE FROM signale")
        db.execute("DELETE FROM analysen")


def als_csv(df: pd.DataFrame, deutsches_format: bool = True, index: bool = True, intraday: bool = False) -> bytes:
    """CSV-Export; deutsches Format = Semikolon und Dezimalkomma (direkt in Excel lesbar)."""
    datumsformat = "%Y-%m-%d %H:%M" if intraday else "%Y-%m-%d"
    daten = df.copy()
    zahlen = daten.select_dtypes(include="number").columns
    daten[zahlen] = daten[zahlen].round(6)
    if deutsches_format:
        return daten.to_csv(sep=";", decimal=",", index=index, date_format=datumsformat).encode("utf-8-sig")
    return daten.to_csv(index=index, date_format=datumsformat).encode("utf-8")


INDIKATOR_EXPORT_SPALTEN = [
    "Close", "SMA_20", "SMA_50", "SMA_200", "EMA_12", "EMA_26", "BB_Oben", "BB_Mitte", "BB_Unten", "BB_ProzentB",
    "BB_Bandbreite", "RSI_14", "MACD", "MACD_Signal", "MACD_Hist", "Stoch_K", "Stoch_D", "ATR_14", "ATR_Prozent",
    "ADX_14", "Plus_DI", "Minus_DI", "OBV", "OBV_SMA_20", "Volumen_SMA_20", "Rel_Volumen", "Volumen_Signal", "Trend_kurz",
    "Trend_mittel", "Trend_lang", "Trendstaerke", *PUNKTE_SPALTEN, "Punkte", "Bewertung", "Kauf_Bestaetigungen",
    "Verkauf_Bestaetigungen", "Signal", "Signaltyp",
]


# =============================================================================
# 14) Streamlit-Oberfläche
# =============================================================================

def _laeuft_in_streamlit() -> bool:
    """True, wenn das Skript über „streamlit run“ ausgeführt wird."""
    if st is None:
        return False
    try:
        from streamlit import runtime
        return bool(runtime.exists())
    except Exception:
        return False


def _cache_data(ttl: int) -> Any:
    """st.cache_data, wenn die App in Streamlit läuft – sonst (Tests, Import) unveränderte Funktion."""
    def dekorator(funktion: Any) -> Any:
        if not _laeuft_in_streamlit():
            return funktion
        return st.cache_data(ttl=ttl, show_spinner=False, max_entries=64)(funktion)
    return dekorator


@_cache_data(ttl=900)
def _daten_laden_gecacht(eingabe: str, zeitraum: str, intervall: str, quelle: str, suffix: str,
                         symbol_unveraendert: bool, dividendenbereinigt: bool, csv_inhalt: bytes | None) -> DatenPaket:
    return daten_laden(eingabe, zeitraum, intervall, quelle, suffix, symbol_unveraendert,
                       dividendenbereinigt, csv_inhalt)


@_cache_data(ttl=3600)
def _wechselkurse_gecacht(ziel: str, start: pd.Timestamp, ende: pd.Timestamp) -> pd.Series:
    return wechselkurse_laden(ziel, start, ende)


@_cache_data(ttl=86400)
def _unternehmensprofil_laden(symbol: str, zeitlimit: float = 8.0) -> dict[str, Any]:
    """Unternehmensprofil von Yahoo Finance (optional). Fehler werden als Ausnahme gemeldet und daher
    nicht zwischengespeichert; ein Zeitlimit verhindert lange Wartezeiten."""
    from concurrent.futures import ThreadPoolExecutor

    import yfinance as yf

    def abrufen() -> dict[str, Any]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return yf.Ticker(symbol).get_info() or {}

    ausfuehrer = ThreadPoolExecutor(max_workers=1)
    try:
        info = ausfuehrer.submit(abrufen).result(timeout=zeitlimit)
    finally:
        ausfuehrer.shutdown(wait=False)
    felder = ("longName", "sector", "industry", "marketCap", "currency", "fullTimeEmployees", "website",
              "longBusinessSummary", "trailingPE", "priceToBook", "dividendYield", "beta")
    profil = {k: info.get(k) for k in felder if info.get(k) not in (None, "")}
    if not profil:
        raise LookupError("Kein Unternehmensprofil verfügbar.")
    return profil


_CSS = """
<style>
.bist-chip {display:inline-block; padding:2px 10px; margin:0 6px 6px 0; border-radius:999px;
            font-size:0.8rem; border:1px solid rgba(128,128,128,0.45);}
.bist-signal {border-radius:12px; padding:14px 16px; border:1px solid; margin-bottom:10px;}
.bist-klein {font-size:0.78rem; opacity:0.8; text-transform:uppercase; letter-spacing:0.04em;}
.bist-titel {font-size:1.35rem; font-weight:700; line-height:1.3;}
.bist-unter {font-size:0.92rem; opacity:0.92; margin-top:2px;}
.bist-stark_pos {background:rgba(21,128,61,0.16); border-color:rgba(21,128,61,0.75);}
.bist-stark_pos .bist-titel {color:#16a34a;}
.bist-pos {background:rgba(22,163,74,0.10); border-color:rgba(22,163,74,0.55);}
.bist-pos .bist-titel {color:#16a34a;}
.bist-neutral {background:rgba(217,119,6,0.10); border-color:rgba(217,119,6,0.55);}
.bist-neutral .bist-titel {color:#d97706;}
.bist-neg {background:rgba(220,38,38,0.10); border-color:rgba(220,38,38,0.55);}
.bist-neg .bist-titel {color:#dc2626;}
.bist-stark_neg {background:rgba(185,28,28,0.16); border-color:rgba(185,28,28,0.75);}
.bist-stark_neg .bist-titel {color:#ef4444;}
[data-testid="stMetricValue"] {font-size:1.55rem;}
[data-testid="stMetricValue"] * {white-space:normal !important; text-overflow:clip !important;
                                   overflow:visible !important;}
[data-testid="stAppDeployButton"] {display:none;}
[data-testid="stMultiSelectTagsContainer"] [role="group"] > span, .stMultiSelect [data-baseweb="tag"]
    {background-color:rgba(100,116,139,0.22) !important; color:inherit !important;}
</style>
"""


def _html(text: Any) -> str:
    import html
    return html.escape(str(text))


def _badge(titel: str, wert: str, unter: str, stil: str) -> str:
    return (f"<div class='bist-signal bist-{stil}'><div class='bist-klein'>{_html(titel)}</div>"
            f"<div class='bist-titel'>{_html(wert)}</div><div class='bist-unter'>{_html(unter)}</div></div>")


def _tabellenhoehe(zeilen: int) -> int:
    """Höhe für st.dataframe, sodass alle Zeilen ohne Scrollen sichtbar sind."""
    return 35 * (int(zeilen) + 1) + 3


def _kennzahl(ziel: Any, titel: str, wert: str, zusatz: str | None = None, info: bool = False,
              hilfe: str | None = None) -> None:
    """st.metric mit Rahmen. info=True: Zusatzzeile neutral (ohne Pfeil und Farbe)."""
    optionen: dict[str, Any] = {"border": True, "help": hilfe}
    if info and zusatz is not None:
        optionen["delta_color"] = "off"
        try:
            import inspect
            if "delta_arrow" in inspect.signature(st.metric).parameters:
                optionen["delta_arrow"] = "off"
        except (TypeError, ValueError):
            pass
    ziel.metric(titel, wert, zusatz, **optionen)


def _farbe_fuer_text(wert: Any) -> str:
    """CSS für Tabellenzellen: Grün = positiv, Rot = negativ, Orange = neutral."""
    stil = SIGNAL_STIL.get(str(wert)) or {"Stark positives Signal": "stark_pos", "Positives Signal": "pos",
                                          "Neutral": "neutral", "Negatives Signal": "neg",
                                          "Stark negatives Signal": "stark_neg"}.get(str(wert))
    if stil is None and str(wert) in STIL_FARBEN:
        stil = str(wert)
    return f"color: {STIL_FARBEN[stil]}; font-weight: 600" if stil else ""


def _farbe_fuer_zahl(wert: Any) -> str:
    if not _ist_zahl(wert) or float(wert) == 0:
        return ""
    return f"color: {FARBEN['positiv'] if float(wert) > 0 else FARBEN['negativ']}; font-weight: 600"


def _startwerte_setzen() -> None:
    """Startwerte der Seitenleiste (auch per URL, z. B. ?ticker=GARAN&quelle=demo)."""
    zustand = st.session_state
    if zustand.get("_initialisiert"):
        return
    parameter = st.query_params
    zustand["ticker"] = parameter.get("ticker", "THYAO")
    quellen = {"yahoo": QUELLE_YAHOO, "borsapy": QUELLE_BORSAPY, "csv": QUELLE_CSV, "demo": QUELLE_DEMO}
    zustand["quelle"] = quellen.get(str(parameter.get("quelle", "")).lower(), QUELLE_YAHOO)
    zustand["zeitraum"] = parameter.get("zeitraum") if parameter.get("zeitraum") in ZEITRAEUME else "1 Jahr"
    zustand["intervall"] = parameter.get("intervall") if parameter.get("intervall") in INTERVALLE else "Täglich"
    zustand["waehrung"] = parameter.get("waehrung") if parameter.get("waehrung") in WAEHRUNGEN else "TRY"
    zustand["schnellauswahl"] = "– eigene Eingabe –"
    zustand["_initialisiert"] = True


def _schnellauswahl_optionen() -> list[str]:
    weitere = [f"{k} – {v}" for k, v in BIST_ALLE_AKTIEN.items() if k not in BIST_FAVORITEN]
    return (["– eigene Eingabe –"] + [f"{k} – {v}" for k, v in BIST_FAVORITEN.items()]
            + [f"{k} – {v} (Index)" for k, v in BIST_INDIZES.items()] + weitere)


def _schnellauswahl_uebernehmen() -> None:
    wahl = st.session_state.get("schnellauswahl", "")
    if wahl and not wahl.startswith("–"):
        st.session_state["ticker"] = wahl.split(" – ")[0].strip()


def _ticker_geaendert() -> None:
    """Manuelle Eingabe → Schnellauswahl zurücksetzen, falls sie nicht mehr passt."""
    wahl = st.session_state.get("schnellauswahl", "")
    eingabe = str(st.session_state.get("ticker", "")).strip().upper()
    if wahl and not wahl.startswith("–") and wahl.split(" – ")[0].strip() != eingabe:
        st.session_state["schnellauswahl"] = "– eigene Eingabe –"


def _gewichte_zuruecksetzen() -> None:
    """Setzt die Punktegewichte auf die Standardwerte zurück."""
    standard = PunkteGewichte()
    for feld in GEWICHT_NAMEN:
        st.session_state[f"gewicht_{feld}"] = getattr(standard, feld)


def _seitenleiste() -> dict[str, Any]:
    """Seitenleiste: Ticker, Zeitraum, Intervall, Indikatoren, Strategie-, Risiko- und Backtest-Parameter."""
    sb = st.sidebar
    sb.header("Einstellungen")
    sb.selectbox("Datenquelle", DATENQUELLEN, key="quelle",
                 help="Yahoo Finance: Standardquelle (BIST-Symbole mit Suffix .IS). borsapy: optionales Paket mit "
                      "TradingView-Daten. CSV: Export Ihres Brokers. Demo: synthetische Daten ohne Internet.")
    quelle = st.session_state["quelle"]
    csv_inhalt = None
    if quelle == QUELLE_CSV:
        datei = sb.file_uploader("CSV-Datei mit Kursdaten", type=["csv", "txt"],
                                 help="Spalten z. B. Datum/Date/Tarih, Eröffnung/Open/Açılış, Hoch/High/Yüksek, "
                                      "Tief/Low/Düşük, Schluss/Close/Kapanış, Volumen/Volume/Hacim. Trennzeichen, "
                                      "Zahlen- und Datumsformat werden automatisch erkannt.")
        csv_inhalt = datei.getvalue() if datei is not None else None
    sb.selectbox("Schnellauswahl", _schnellauswahl_optionen(), key="schnellauswahl",
                 on_change=_schnellauswahl_uebernehmen,
                 help=f"Große Werte und Indizes zuerst, danach alle {len(BIST_ALLE_AKTIEN)} BIST-Aktien. "
                      "Zum Suchen einfach tippen.")
    sb.text_input("Ticker (Börsenkürzel)", key="ticker", placeholder="z. B. THYAO", on_change=_ticker_geaendert,
                  help="Beispiele: THYAO, GARAN, ASELS oder XU100 (BIST 100). Formate wie THYAO.IS, BIST:THYAO, "
                       "THYAO.E oder ŞİŞE werden automatisch umgewandelt. Bei CSV/Demo dient der Ticker als Name.")
    sb.selectbox("Zeitraum", list(ZEITRAEUME), key="zeitraum")
    sb.selectbox("Intervall", list(INTERVALLE), key="intervall",
                 help="Intraday-Daten: bei Yahoo Finance max. 60 Tage (60 Min.: ca. 2 Jahre) Historie.")
    sb.radio("Währung", list(WAEHRUNGEN), key="waehrung", horizontal=True,
             help="Umrechnung von TRY-Kursen mit Tages-Wechselkursen (Yahoo Finance). Wegen der hohen "
                  "TRY-Inflation zeigt die USD/EUR-Sicht die reale Entwicklung oft besser.")

    with sb.expander("Chart & Indikatoren"):
        overlays = st.multiselect("Im Kurschart", OVERLAY_OPTIONEN, default=OVERLAY_STANDARD)
        panels = st.multiselect("Indikator-Fenster", PANEL_OPTIONEN, default=PANEL_STANDARD)

    with sb.expander("Strategieparameter"):
        rsi_ov, rsi_ob = st.slider("RSI: überverkauft / überkauft", 5, 95, (30, 70),
                                   help="RSI unter dem linken Wert gilt als überverkauft, über dem rechten als überkauft.")
        rsi_kauf = st.slider("RSI-Obergrenze für Kaufsignale", 40, 95, 65,
                             help="Kaufsignale nur, wenn der RSI zwischen „überverkauft“ und diesem Wert liegt.")
        adx = st.slider("ADX-Schwelle (Trendstärke)", 10, 50, 25)
        vol_faktor = st.slider("Volumen-Faktor (× Ø 20)", 0.5, 3.0, 1.2, 0.1,
                               help="Volumen gilt ab diesem Vielfachen des 20-Perioden-Durchschnitts als erhöht.")
        macd_fenster = st.slider("Signalfenster MACD-Kreuzung (Kerzen)", 1, 10, 3,
                                 help="So viele Kerzen nach einer MACD-Kreuzung kann ein Signal entstehen; "
                                      "gilt auch für das Volumen-Signal der Punktbewertung.")
        sma_fenster = st.slider("Signalfenster Golden/Death Cross (Kerzen)", 1, 60, 10,
                                help="So viele Kerzen zählt eine Kreuzung von SMA 50 und SMA 200 in der Bewertung.")
        min_best = st.slider("Mindestanzahl Bestätigungen (von 6)", 1, 6, 4,
                             help="Zusatzbedingungen, die neben dem MACD-Auslöser erfüllt sein müssen.")
        st.markdown("**Bewertungsschwellen (Punkte)**")
        neg, pos = st.slider("negativ ab / positiv ab", -12, 12, (-3, 3),
                             help="Punktzahlen zwischen den beiden Werten gelten als neutral.")
        stark_neg, stark_pos = st.slider("stark negativ ab / stark positiv ab", -15, 15, (-6, 6))

    with sb.expander("Punktegewichtung"):
        std = PunkteGewichte()
        werte = {feld: st.number_input(name, -5, 5, getattr(std, feld), 1, key=f"gewicht_{feld}")
                 for feld, name in GEWICHT_NAMEN.items()}
        gewichte = PunkteGewichte(**{feld: int(wert) for feld, wert in werte.items()})
        st.caption("Gewicht 0 deaktiviert eine Komponente.")
        st.button("Standardgewichte wiederherstellen", icon=":material/restart_alt:", on_click=_gewichte_zuruecksetzen)

    with sb.expander("Risiko & Backtest"):
        kapital = st.number_input("Kapital / Depotgröße", 1_000.0, 1e11, 100_000.0, 10_000.0, format="%.0f")
        risiko = st.number_input("Max. Risiko pro Position (%)", 0.1, 20.0, 1.0, 0.1)
        atr_mult = st.number_input("Stop-Loss: ATR-Multiplikator", 0.5, 10.0, 2.0, 0.1)
        crv = st.number_input("Chance-Risiko-Verhältnis (CRV)", 0.5, 10.0, 2.0, 0.1)
        kosten = st.number_input("Transaktionskosten je Order (%)", 0.0, 3.0, 0.10, 0.01, format="%.2f",
                                 help="Provision inkl. Gebühren/Steuern auf die Provision, je Kauf bzw. Verkauf.")
        slippage = st.number_input("Slippage je Ausführung (%)", 0.0, 3.0, 0.05, 0.01, format="%.2f")
        zins = st.number_input("Risikofreier Zins p. a. (%)", 0.0, 100.0, 0.0, 0.5,
                               help="Für die Sharpe Ratio. Bei TRY-Betrachtung den aktuellen TRY-Zins (z. B. "
                                    "Leitzins der TCMB) eintragen – sonst wirkt die Sharpe Ratio zu günstig.")
        st.markdown("**Backtest**")
        stop_aktiv = st.checkbox("ATR-Stop-Loss verwenden", value=True)
        ziel_aktiv = st.checkbox("Kursziel (CRV) verwenden", value=True)
        nur_stark = st.checkbox("Nur starke Kaufsignale handeln", value=False)
        groesse = st.radio("Positionsgröße", ["Volles Kapital", "Risikobasiert"], horizontal=True,
                           help="Risikobasiert: Stückzahl so, dass ein Stop-Loss höchstens das eingestellte "
                                "Risiko pro Position kostet.")

    with sb.expander("Ticker-Format & Daten"):
        suffix = st.text_input("Börsen-Suffix (Yahoo Finance)", STANDARD_SUFFIX,
                               help="Borsa İstanbul = .IS. Andere Anbieter nutzen z. B. BIST:THYAO (TradingView) "
                                    "oder THYAO.E (Foreks/Matriks).")
        unveraendert = st.checkbox("Symbol unverändert verwenden", value=False,
                                   help="Kein Suffix ergänzen – z. B. für andere Börsen (AAPL, SAP.DE).")
        dividende = st.checkbox("Dividendenbereinigte Kurse", value=True,
                                help="Yahoo Finance: rückwirkende Bereinigung um Dividenden (Total-Return-Sicht). "
                                     "Splits/Bonusaktien werden immer bereinigt.")
        profil = st.checkbox("Unternehmensprofil laden", value=True,
                             help="Zusätzliche Abfrage bei Yahoo Finance (Branche, Marktkapitalisierung).")
        if st.button("Daten neu laden (Cache leeren)", icon=":material/refresh:"):
            st.cache_data.clear()

    sb.caption(f"{APP_NAME} v{APP_VERSION}")
    sp = StrategieParameter(
        rsi_ueberverkauft=float(rsi_ov), rsi_ueberkauft=float(rsi_ob), rsi_kauf_obergrenze=float(rsi_kauf),
        adx_schwelle=float(adx), volumen_faktor=float(vol_faktor), macd_fenster=int(macd_fenster),
        sma_kreuz_fenster=int(sma_fenster), min_bestaetigungen=int(min_best),
        schwelle_stark_positiv=int(stark_pos), schwelle_positiv=int(pos), schwelle_negativ=int(neg),
        schwelle_stark_negativ=int(stark_neg), gewichte=gewichte)
    rp = RisikoParameter(kapital=float(kapital), risiko_pro_position_pct=float(risiko),
                         atr_multiplikator=float(atr_mult), crv=float(crv), transaktionskosten_pct=float(kosten),
                         slippage_pct=float(slippage), risikofreier_zins_pct=float(zins))
    bp = BacktestParameter(stop_loss_aktiv=stop_aktiv, kursziel_aktiv=ziel_aktiv, nur_starke_signale=nur_stark,
                           positionsgroesse="risiko" if groesse == "Risikobasiert" else "voll")
    return {
        "quelle": quelle, "csv_inhalt": csv_inhalt, "ticker": st.session_state.get("ticker", ""),
        "zeitraum": st.session_state["zeitraum"], "intervall": st.session_state["intervall"],
        "waehrung": st.session_state["waehrung"], "overlays": overlays, "panels": panels,
        "sp": sp, "rp": rp, "bp": bp, "suffix": suffix, "unveraendert": unveraendert,
        "dividende": dividende, "profil": profil,
    }


def _fehler_anzeigen(fehler: DatenFehler) -> None:
    titel = {"keine_daten": "Keine Daten gefunden", "netzwerk": "Verbindungsproblem", "limit": "Zu viele Anfragen",
             "eingabe": "Eingabe prüfen", "abhaengigkeit": "Fehlendes Paket"}.get(fehler.art, "Fehler")
    st.error(f"**{titel}:** {fehler.meldung}", icon=":material/error:")
    if fehler.tipps:
        st.info("**Hinweise:**\n\n" + "\n".join(f"- {t}" for t in fehler.tipps), icon=":material/lightbulb:")


def kurs_einheit(paket: DatenPaket, waehrung: str) -> str:
    """Einheit für Kursangaben: Indizes in Punkten, Aktien in ihrer Währung."""
    if not paket.ist_index:
        return waehrung
    return "Pkt." if waehrung == paket.waehrung else f"Pkt. ({waehrung})"


def _kopfbereich(paket: DatenPaket, uebersicht: dict[str, Any], waehrung: str, info: IntervallInfo,
                 zeige_profil: bool) -> None:
    """Unternehmens- und Kursübersicht."""
    st.subheader(f"{paket.name} · {paket.symbol}")
    art = "Index" if paket.ist_index else "Aktie"
    boerse = paket.meta.get("fullExchangeName") or paket.meta.get("exchangeName") or "Borsa İstanbul"
    chips = [art, str(boerse), f"Quelle: {paket.quelle}", f"Währung: {waehrung}", f"Intervall: {paket.intervall}",
             f"Zeitraum: {paket.zeitraum}"]
    st.markdown(" ".join(f"<span class='bist-chip'>{_html(c)}</span>" for c in chips), unsafe_allow_html=True)

    einheit = kurs_einheit(paket, waehrung)
    spalten = st.columns(5)
    delta = None
    if _ist_zahl(uebersicht["veraenderung"]):
        delta = f"{fmt_zahl(uebersicht['veraenderung'], 2, True)} ({fmt_pct(uebersicht['veraenderung_pct'])})"
    spalten[0].metric("Letzter Kurs" if not paket.ist_index else "Indexstand",
                      f"{fmt_zahl(uebersicht['kurs'])} {einheit}", delta, border=True,
                      help="Veränderung gegenüber dem Schlusskurs des vorherigen Handelstags.")
    spalten[1].metric("Volumen (Stück)", fmt_volumen(uebersicht["volumen"]), border=True,
                      help="Gehandelte Stückzahl am letzten Handelstag.")
    if paket.ist_index:
        spalten[2].metric("Tagesspanne", fmt_spanne(uebersicht["tagestief"], uebersicht["tageshoch"]), border=True,
                          help="Tagestief und Tageshoch des letzten Handelstags.")
    else:
        spalten[2].metric("Umsatz (ca.)", f"{fmt_volumen(uebersicht['umsatz'])} {waehrung}", border=True,
                          help="Volumen × letzter Kurs (Näherung).")
    hoch, tief = uebersicht["hoch_52w"], uebersicht["tief_52w"]
    if not _ist_zahl(hoch) and waehrung == paket.waehrung:
        hoch, tief = paket.meta.get("fiftyTwoWeekHigh"), paket.meta.get("fiftyTwoWeekLow")
    spalten[3].metric("52-Wochen-Spanne", fmt_spanne(tief, hoch), border=True,
                      help="Tiefst- und Höchstkurs der letzten 52 Wochen.")
    spalten[4].metric("Datenstand", fmt_datum(uebersicht["datum"], info.intraday), border=True,
                      help="Zeitpunkt der letzten Kerze (Börsenzeit Istanbul). Kurse können verzögert sein.")

    if zeige_profil and paket.quelle == QUELLE_YAHOO and not paket.ist_index:
        try:
            profil = _unternehmensprofil_laden(paket.symbol)
        except Exception:   # Profil ist optional (Zeitlimit, Rate-Limit, fehlende Daten)
            profil = {}
        with st.expander("Unternehmensprofil"):
            if not profil:
                st.caption("Unternehmensprofil derzeit nicht verfügbar.")
            else:
                a, b, c = st.columns(3)
                a.markdown(f"**Sektor:** {profil.get('sector', '–')}  \n**Branche:** {profil.get('industry', '–')}")
                b.markdown(f"**Marktkapitalisierung:** {fmt_volumen(profil.get('marketCap'))} "
                           f"{profil.get('currency', '')}  \n**Mitarbeiter:** {fmt_zahl(profil.get('fullTimeEmployees'), 0)}")
                c.markdown(f"**KGV:** {fmt_zahl(profil.get('trailingPE'))}  \n**KBV:** {fmt_zahl(profil.get('priceToBook'))}")
                if profil.get("website"):
                    st.markdown(f"**Website:** {profil['website']}")
                if profil.get("longBusinessSummary"):
                    st.caption(str(profil["longBusinessSummary"])[:900])


def _gesamtsignal_bereich(erg: AnalyseErgebnis, kurzfazit: str, waehrung: str) -> None:
    """Aktuelles Gesamtsignal: Punkte-Tachometer, Bewertung, Signalstatus, Stop/Ziel und Kurzfazit."""
    st.markdown("### Aktuelles Gesamtsignal")
    z = erg.d_voll.iloc[-1]
    punkte = int(z["Punkte"])
    links, mitte, rechts = st.columns([1.05, 1.6, 1.25])
    with links:
        st.plotly_chart(punkte_anzeige_erstellen(punkte, erg.sp), config={"displayModeBar": False},
                        key="punkte_anzeige")
        st.caption(f"Gesamtpunktzahl (möglich: {erg.sp.gewichte.minimum():+d} bis {erg.sp.gewichte.maximum():+d})")
    with mitte:
        bewertung, stil = erg.bewertung
        st.markdown(_badge("Technische Bewertung", bewertung, f"{punkte_text(punkte)} · Schwellen: stark positiv ≥ "
                           f"{erg.sp.schwelle_stark_positiv}, positiv ≥ {erg.sp.schwelle_positiv}, negativ ≤ "
                           f"{erg.sp.schwelle_negativ}, stark negativ ≤ {erg.sp.schwelle_stark_negativ}", stil),
                    unsafe_allow_html=True)
        status = erg.status
        if status["code"] != 0:
            alter = "auf der letzten Kerze" if status["alter"] == 0 else f"vor {status['alter']} Kerze(n)"
            unter = f"Signal {alter} · {fmt_datum(status['datum'], erg.intervall.intraday)}"
        else:
            letztes = erg.signale.iloc[-1] if not erg.signale.empty else None
            unter = ("Kein frisches Signal · letztes Signal: "
                     + (f"{letztes['Signaltyp']} am {fmt_datum(letztes['Datum'], erg.intervall.intraday)}"
                        if letztes is not None else "keines im Zeitraum"))
        if erg.letzte_kerze_vorlaeufig:
            unter += " · letzte Kerze vorläufig"
        st.markdown(_badge("Signalstatus", status["status"], unter, SIGNAL_STIL[status["status"]]),
                    unsafe_allow_html=True)
    with rechts:
        r = erg.risiko
        st.metric(f"Stop-Loss ({fmt_zahl(erg.rp.atr_multiplikator, 1)} × ATR)", f"{fmt_zahl(r.stop_loss)} {waehrung}",
                  fmt_pct(r.stop_abstand) if _ist_zahl(r.stop_abstand) else None, border=True)
        st.metric(f"Kursziel (CRV {fmt_zahl(erg.rp.crv, 1)})", f"{fmt_zahl(r.kursziel)} {waehrung}",
                  fmt_pct(r.ziel_abstand) if _ist_zahl(r.ziel_abstand) else None, border=True)
    st.info(f"**Zusammenfassung:** {kurzfazit}", icon=":material/summarize:")


def _tabelle_signale(signale: pd.DataFrame, intraday: bool) -> Any:
    ansicht = signale.drop(columns=["Code", "Stärke"], errors="ignore").iloc[::-1].reset_index(drop=True)
    return (ansicht.style
            .map(_farbe_fuer_text, subset=["Signaltyp"])
            .map(_farbe_fuer_zahl, subset=["Punktzahl"])
            .format({"Datum": lambda t: fmt_datum(t, intraday), "Kurs": lambda v: fmt_zahl(v),
                     "Punktzahl": fmt_punkte}))


def _tab_chart(erg: AnalyseErgebnis, e: dict[str, Any], titel: str, waehrung: str) -> go.Figure:
    panels = list(e["panels"])
    if not erg.volumen_verfuegbar:
        panels = [p for p in panels if p not in ("Volumen", "OBV")]
    fig = chart_erstellen(erg.d, erg.signale, erg.zonen, e["overlays"], panels, erg.sp, "", erg.intervall, waehrung)
    st.plotly_chart(fig, config={"scrollZoom": True, "displaylogo": False,
                                 "modeBarButtonsToRemove": ["lasso2d", "select2d"]}, key="kurschart")
    fig = go.Figure(fig).update_layout(title=dict(text=titel, x=0, y=0.995, yanchor="top", font=dict(size=15)),
                                       margin=dict(t=110))   # Variante mit Titel für den HTML-Export
    st.caption("Legende anklicken = Element ein-/ausblenden, Doppelklick = nur dieses Element. Ziehen zoomt, "
               "Doppelklick in den Chart setzt zurück. ▲ Kauf-, ▼ Verkaufssignal (Stern = starkes Signal). "
               "Grüne/rote Flächen: Unterstützungs-/Widerstandszonen (U/W, Anzahl Berührungen).")
    if not erg.volumen_verfuegbar:
        st.caption("Volumen und OBV werden ausgeblendet, da keine verlässlichen Volumendaten vorliegen.")
    return fig


def _tab_indikatoren(erg: AnalyseErgebnis) -> None:
    eintraege = pd.DataFrame(indikator_uebersicht(erg.d_voll, erg.sp, erg.volumen_verfuegbar))
    stile = dict(zip(eintraege["Einordnung"], eintraege["Stil"], strict=True))
    tabelle = eintraege.drop(columns=["Stil"]).style.map(lambda v: _farbe_fuer_text(stile.get(v, "")),
                                                         subset=["Einordnung"])
    st.dataframe(tabelle, hide_index=True, height=_tabellenhoehe(len(eintraege)))
    with st.expander("Indikatorwerte der letzten 30 Kerzen"):
        spalten = ["Close", "SMA_20", "SMA_50", "SMA_200", "EMA_12", "EMA_26", "RSI_14", "MACD", "MACD_Signal",
                   "Stoch_K", "Stoch_D", "ATR_14", "ADX_14", "OBV", "Rel_Volumen", "Punkte"]
        ansicht = erg.d[spalten].tail(30).iloc[::-1]
        st.dataframe(ansicht.style.format(lambda v: fmt_zahl(v, 2), subset=[s for s in spalten if s != "Punkte"])
                     .format_index(lambda t: fmt_datum(t, erg.intervall.intraday)))


def _tab_begruendung(erg: AnalyseErgebnis) -> None:
    st.markdown("#### Trendanalyse")
    spalten = st.columns(4)
    for spalte, (schluessel, titel) in zip(spalten, (("kurz", "Kurzfristig"), ("mittel", "Mittelfristig"),
                                                     ("lang", "Langfristig")), strict=False):
        horizont = erg.trend["horizonte"][schluessel]
        farbe = {1: "green", -1: "red", 0: "orange"}.get(horizont["wert"], "gray")
        spalte.markdown(f"**{titel}**  \n:{farbe}[**{horizont['text']}**]  \n<small>{horizont['grundlage']}</small>",
                        unsafe_allow_html=True)
    spalten[3].markdown(f"**Trendstärke**  \n{erg.trend['staerke']}  \n<small>{erg.trend['richtung']}</small>",
                        unsafe_allow_html=True)

    st.markdown("#### Punktbewertung der letzten Kerze")
    tabelle = erg.punkte_tabelle.style.map(_farbe_fuer_zahl, subset=["Punkte"]).format(
        {"Punkte": fmt_punkte})
    st.dataframe(tabelle, hide_index=True, height=_tabellenhoehe(len(erg.punkte_tabelle)))
    st.markdown(f"**Summe: {punkte_text(erg.d_voll['Punkte'].iloc[-1])} → {erg.bewertung[0]}**")

    st.markdown("#### Signalbedingungen (letzte Kerze)")
    z = erg.d_voll.iloc[-1]
    links, rechts = st.columns(2)
    for spalte, titel, ausloeser, bedingungen in (
            (links, "Kaufsignal", bool(z["MACD_Kaufimpuls"]), erg.kauf_bedingungen),
            (rechts, "Verkaufssignal", bool(z["MACD_Verkaufsimpuls"]), erg.verkauf_bedingungen)):
        richtung = "nach oben" if titel == "Kaufsignal" else "nach unten"
        erfuellt = sum(bedingungen.values())
        zeilen = [f"**{titel}** – Auslöser: {':green[✓]' if ausloeser else ':red[✗]'} MACD-Kreuzung {richtung} "
                  f"(innerhalb {erg.sp.macd_fenster} Kerzen)", ""]
        zeilen += [f"- {':green[✓]' if ok else ':red[✗]'} {name}" for name, ok in bedingungen.items()]
        zeilen += ["", f"Erfüllt: **{erfuellt} von {len(bedingungen)}** (benötigt: "
                       f"{min(erg.sp.min_bestaetigungen, len(bedingungen))})"]
        spalte.markdown("\n".join(zeilen))
    st.caption(f"Ein Kaufsignal entsteht nur, wenn der Auslöser aktiv ist, mindestens {erg.sp.min_bestaetigungen} "
               f"Bestätigungen erfüllt sind und die Gesamtpunktzahl mindestens {erg.sp.schwelle_positiv} beträgt "
               f"(stark ab {erg.sp.schwelle_stark_positiv}). Verkaufssignale entsprechend spiegelbildlich "
               f"(≤ {erg.sp.schwelle_negativ}, stark ≤ {erg.sp.schwelle_stark_negativ}). Pro MACD-Kreuzung wird "
               "höchstens ein Signal erzeugt.")


def _tab_risiko(erg: AnalyseErgebnis, waehrung: str, einheit: str | None = None) -> None:
    einheit = einheit or waehrung
    r, rp, intraday = erg.risiko, erg.rp, erg.intervall.intraday
    a, b, c, d = st.columns(4)
    _kennzahl(a, "ATR 14", f"{fmt_zahl(r.atr)} {einheit}", f"{fmt_pct(r.atr_prozent, 2, False)} des Kurses",
              info=True)
    b.metric(f"Stop-Loss ({fmt_zahl(rp.atr_multiplikator, 1)} × ATR)", f"{fmt_zahl(r.stop_loss)} {einheit}",
             fmt_pct(r.stop_abstand) if _ist_zahl(r.stop_abstand) else None, border=True)
    c.metric(f"Kursziel (CRV {fmt_zahl(rp.crv, 1)})", f"{fmt_zahl(r.kursziel)} {einheit}",
             fmt_pct(r.ziel_abstand) if _ist_zahl(r.ziel_abstand) else None, border=True)
    _kennzahl(d, "Positionsgröße", f"{r.stueckzahl} Stück",
              f"≈ {fmt_zahl(r.positionswert, 0)} {waehrung} · {fmt_pct(r.positionsanteil, 1, False)}", info=True,
              hilfe=f"Kapital {fmt_zahl(rp.kapital, 0)} {waehrung}, Risiko {fmt_zahl(rp.risiko_pro_position_pct, 1)} % "
                    f"je Position, inkl. Kosten und Slippage. Prozentwert = Anteil am Kapital.")
    a, b, c, d = st.columns(4)
    _kennzahl(a, "Volatilität p. a.", fmt_pct(r.volatilitaet_pa, 1, False),
              f"letzte 20 Kerzen: {fmt_pct(r.volatilitaet_20_pa, 1, False)}", info=True)
    b.metric("VaR 95 % (1 Kerze)", fmt_pct(-r.var_95 if _ist_zahl(r.var_95) else None), border=True,
             help="Historischer Value at Risk: In 95 % der Kerzen war der Verlust nicht größer.")
    c.metric("Max. Drawdown", fmt_pct(r.max_drawdown), border=True,
             help="Größter Rückgang vom Hoch zum Tief im Anzeigezeitraum.")
    d.metric("Sharpe Ratio (Buy & Hold)", fmt_zahl(r.sharpe, 2) if r.sharpe is not None else "–", border=True,
             help=r.sharpe_hinweis or f"Annualisiert, risikofreier Zins {fmt_zahl(rp.risikofreier_zins_pct, 1)} % p. a.")
    st.markdown(
        f"- **Maximaler Verlust bei Stop-Loss:** ca. {fmt_zahl(r.max_verlust, 0)} {waehrung} "
        f"(Risiko je Aktie {fmt_zahl(r.risiko_je_aktie)} inkl. Kosten)\n"
        f"- **Längste Verlustserie:** {r.verlustserie_laenge} Kerzen in Folge mit fallendem Schlusskurs "
        f"({fmt_pct(r.verlustserie_verlust)}, {fmt_datum(r.verlustserie_start, intraday)} – "
        f"{fmt_datum(r.verlustserie_ende, intraday)})\n"
        f"- **Maximale Drawdown-Phase:** {fmt_pct(r.max_drawdown)} vom Hoch am {fmt_datum(r.dd_hoch, intraday)} bis "
        f"zum Tief am {fmt_datum(r.dd_tief, intraday)}; "
        + (f"erholt am {fmt_datum(r.dd_erholung, intraday)}" if r.dd_erholung is not None else "noch nicht erholt")
        + f" (Dauer {r.dd_dauer_kerzen} Kerzen / {r.dd_dauer_tage} Kalendertage)\n"
        f"- **Nächste Unterstützung:** "
        + (f"{fmt_zahl(r.naechste_unterstuetzung.unten)} – {fmt_zahl(r.naechste_unterstuetzung.oben)} "
           f"({r.naechste_unterstuetzung.beruehrungen} Berührungen)" if r.naechste_unterstuetzung else "keine erkannt")
        + "\n- **Nächster Widerstand:** "
        + (f"{fmt_zahl(r.naechster_widerstand.unten)} – {fmt_zahl(r.naechster_widerstand.oben)} "
           f"({r.naechster_widerstand.beruehrungen} Berührungen)" if r.naechster_widerstand else "keiner erkannt"))
    for hinweis in r.hinweise:
        st.warning(hinweis, icon=":material/warning:")
    st.caption("Long-Betrachtung: Stop-Loss = Kurs − Multiplikator × ATR 14, Kursziel = Kurs + CRV × (Kurs − Stop). "
               "Leerverkäufe sind an der Borsa İstanbul für Privatanleger nur eingeschränkt möglich.")


def _tab_backtest(erg: AnalyseErgebnis, waehrung: str) -> None:
    st.warning(BACKTEST_HINWEIS, icon=":material/history:")
    bt = erg.backtest
    if bt is None:
        st.info("Für den Backtest liegen nicht genügend Daten vor.")
        return
    for hinweis in bt.hinweise:
        st.info(hinweis)
    s, b = bt.werte["Strategie"], bt.werte["Buy & Hold"]
    a, b1, c, d = st.columns(4)
    profit_faktor = s["Profit Factor"]
    _kennzahl(a, "Gesamtrendite Strategie", fmt_pct(s["Gesamtrendite"]),
              f"Buy & Hold: {fmt_pct(b['Gesamtrendite'])}", info=True)
    _kennzahl(b1, "Annualisierte Rendite", fmt_pct(s["Annualisierte Rendite"]),
              f"Buy & Hold: {fmt_pct(b['Annualisierte Rendite'])}", info=True)
    _kennzahl(c, "Max. Drawdown", fmt_pct(s["Maximaler Drawdown"]),
              f"Buy & Hold: {fmt_pct(b['Maximaler Drawdown'])}", info=True)
    _kennzahl(d, "Trades / Trefferquote", f"{s['Anzahl Trades']} / {fmt_pct(s['Trefferquote'], 0, False)}",
              f"Profit Factor: {'∞' if profit_faktor == float('inf') else fmt_zahl(profit_faktor)}", info=True)
    st.plotly_chart(backtest_chart_erstellen(bt, waehrung, erg.intervall.intraday), key="backtest_chart")
    links, _ = st.columns([1.2, 1])
    with links:
        st.markdown("#### Kennzahlen")
        st.dataframe(bt.kennzahlen, hide_index=True, height=_tabellenhoehe(len(bt.kennzahlen)))
    st.markdown("#### Trades")
    if bt.trades.empty:
        st.caption("Keine Trades im Zeitraum.")
    else:
        intraday = erg.intervall.intraday
        tabelle = (bt.trades.iloc[::-1].reset_index(drop=True).style
                   .map(_farbe_fuer_zahl, subset=["Ergebnis", "Rendite"])
                   .format({"Einstieg": lambda t: fmt_datum(t, intraday), "Ausstieg": lambda t: fmt_datum(t, intraday),
                            "Einstiegskurs": fmt_zahl, "Ausstiegskurs": fmt_zahl, "Stop-Loss": fmt_zahl,
                            "Kursziel": fmt_zahl, "Ergebnis": lambda v: fmt_zahl(v, 0, True),
                            "Rendite": lambda v: fmt_pct(v), "Kosten": lambda v: fmt_zahl(v, 0)}))
        st.dataframe(tabelle, hide_index=True)
    with st.expander("Methodik und Annahmen"):
        st.markdown(
            "- **Kein Look-Ahead-Bias:** Signale werden mit dem Schlusskurs einer Kerze berechnet und erst zum "
            "Eröffnungskurs der **folgenden** Kerze ausgeführt. Alle Indikatoren nutzen nur vergangene Daten.\n"
            "- **Stop-Loss/Kursziel:** Einstieg − Multiplikator × ATR der Signalkerze bzw. Einstieg + CRV × Risiko; "
            "Prüfung innerhalb der Kerze. Bei Kurslücken wird zum Eröffnungskurs ausgeführt; werden Stop und Ziel in "
            "derselben Kerze erreicht, wird konservativ der Stop angenommen.\n"
            "- **Ausstieg** zudem bei einem Verkaufssignal (zur nächsten Eröffnung); eine am Ende offene Position "
            "wird zum letzten Schlusskurs inkl. Kosten bewertet.\n"
            "- **Kosten:** Transaktionskosten je Kauf und Verkauf, Slippage verschlechtert jeden Ausführungskurs.\n"
            "- **Nur Long**, ganze Stückzahlen, keine Steuern, keine Dividendenzahlungen (bei dividendenbereinigten "
            "Kursen sind sie im Kursverlauf enthalten).\n"
            "- **Grenzen:** Rückwirkend bereinigte Kurse, Survivorship-Bias (nur heute gelistete Werte) und "
            "Datenfehler können Ergebnisse verzerren. Optimierte Parameter führen leicht zu Überanpassung.")


def _tab_signalhistorie(erg: AnalyseErgebnis) -> None:
    if erg.signale.empty:
        st.info("Im gewählten Zeitraum wurden keine Kauf- oder Verkaufssignale erzeugt.")
        return
    anzahl = erg.signale["Signaltyp"].value_counts()
    spalten = st.columns(4)
    for spalte, typ in zip(spalten, (KAUF_STARK, KAUF, VERKAUF, VERKAUF_STARK), strict=True):
        spalte.metric(typ, int(anzahl.get(typ, 0)), border=True)
    auswahl = st.multiselect("Signaltypen filtern", [KAUF_STARK, KAUF, VERKAUF, VERKAUF_STARK],
                             default=[KAUF_STARK, KAUF, VERKAUF, VERKAUF_STARK])
    gefiltert = erg.signale[erg.signale["Signaltyp"].isin(auswahl)]
    st.dataframe(_tabelle_signale(gefiltert, erg.intervall.intraday), hide_index=True,
                 column_order=["Datum", "Kurs", "Signaltyp", "Punktzahl", "Signalstärke", "Auslösende Indikatoren",
                               "Erklärung"])
    with st.expander("Erklärungen der Signale im Detail", expanded=False):
        for _, s in gefiltert.iloc[::-1].iterrows():
            farbe = "green" if s["Code"] > 0 else "red"
            st.markdown(f"**{fmt_datum(s['Datum'], erg.intervall.intraday)} · :{farbe}[{s['Signaltyp']}]** · "
                        f"Kurs {fmt_zahl(s['Kurs'])} · {punkte_text(s['Punktzahl'])} · Stärke {s['Signalstärke']}  \n"
                        f"{s['Erklärung']}")
    if erg.letzte_kerze_vorlaeufig and not erg.signale.empty and erg.signale["Datum"].iloc[-1] == erg.d.index[-1]:
        st.caption("Das jüngste Signal basiert auf einer noch nicht abgeschlossenen Kerze und ist vorläufig.")


def _tab_export(erg: AnalyseErgebnis, paket: DatenPaket, fig: go.Figure | None, bericht: str, waehrung: str) -> None:
    with st.expander("Bericht anzeigen", expanded=False):
        st.markdown(bericht)
    format_wahl = st.radio("CSV-Format", ["Deutsch/Excel (Semikolon, Dezimalkomma)",
                                          "International (Komma, Dezimalpunkt)"], horizontal=True)
    deutsch = format_wahl.startswith("Deutsch")
    intraday = erg.intervall.intraday
    kennung = re.sub(r"[^A-Za-z0-9_-]", "_", paket.symbol)
    stand = pd.Timestamp(erg.d.index[-1]).strftime("%Y%m%d")

    def knopf(spalte: Any, text: str, daten: bytes | str, datei: str, mime: str = "text/csv") -> None:
        stamm, endung = datei.rsplit(".", 1)
        spalte.download_button(text, daten, file_name=f"{kennung}_{stamm}_{stand}.{endung}", mime=mime,
                               on_click="ignore", icon=":material/download:")

    a, b, c = st.columns(3)
    kurse = erg.d[[s for s in OHLCV if s in erg.d]].copy()
    kurse["Volume"] = kurse["Volume"].round().astype("int64")
    if deutsch:
        kurse = kurse.rename(columns={"Open": "Eröffnung", "High": "Hoch", "Low": "Tief", "Close": "Schluss",
                                      "Volume": "Volumen"})
    knopf(a, "Historische Kursdaten (CSV)", als_csv(kurse, deutsch, intraday=intraday), "kursdaten.csv")
    indikatoren = erg.d[[s for s in INDIKATOR_EXPORT_SPALTEN if s in erg.d]]
    knopf(b, "Technische Indikatoren (CSV)", als_csv(indikatoren, deutsch, intraday=intraday), "indikatoren.csv")
    signale = erg.signale.drop(columns=["Code"], errors="ignore")
    knopf(c, "Signalhistorie (CSV)", als_csv(signale, deutsch, index=False, intraday=intraday), "signale.csv")
    if erg.backtest is not None:
        a, b, c = st.columns(3)
        werte = pd.DataFrame(erg.backtest.werte).rename_axis("Kennzahl")
        knopf(a, "Backtest-Kennzahlen (CSV)", als_csv(werte, deutsch), "backtest_kennzahlen.csv")
        knopf(b, "Backtest-Trades (CSV)", als_csv(erg.backtest.trades, deutsch, index=False, intraday=intraday),
              "backtest_trades.csv")
        knopf(c, "Kapitalverlauf (CSV)", als_csv(erg.backtest.verlauf, deutsch, intraday=intraday),
              "backtest_kapitalverlauf.csv")
    a, b, _ = st.columns(3)
    if fig is not None:
        knopf(a, "Chart (HTML, interaktiv)", fig.to_html(include_plotlyjs="cdn", full_html=True), "chart.html",
              "text/html")
    knopf(b, "Bericht (Markdown)", bericht, "bericht.md", "text/markdown")
    st.caption("Die HTML-Datei lädt Plotly beim Öffnen aus dem Internet (CDN). CSV im deutschen Format lässt sich "
               "direkt in Excel öffnen.")


def _tab_datenbank(erg: AnalyseErgebnis, paket: DatenPaket, bericht: str, waehrung: str) -> None:
    st.markdown("#### Analysen speichern (SQLite)")
    st.caption(f"Datenbankdatei: {DB_DATEI}")
    if st.button("Aktuelle Analyse speichern", icon=":material/save:"):
        try:
            nummer = analyse_speichern(erg, paket.name, paket.symbol, paket.quelle, paket.zeitraum, paket.intervall,
                                       waehrung, bericht)
            st.success(f"Analyse Nr. {nummer} mit {len(erg.signale)} Signalen gespeichert.")
        except Exception as exc:  # z. B. schreibgeschütztes Verzeichnis
            st.error(f"Speichern fehlgeschlagen: {type(exc).__name__}: {exc}")
    try:
        verlauf = analysen_laden()
    except Exception as exc:
        st.error(f"Datenbank konnte nicht gelesen werden: {exc}")
        return
    if verlauf.empty:
        st.caption("Noch keine gespeicherten Analysen.")
        return
    st.dataframe(verlauf.style.map(_farbe_fuer_text, subset=["bewertung", "signalstatus"])
                 .format({"kurs": fmt_zahl, "stop_loss": fmt_zahl, "kursziel": fmt_zahl,
                          "rendite_strategie": fmt_pct, "rendite_buy_hold": fmt_pct}), hide_index=True)
    a, b = st.columns([2, 1])
    nummer = a.selectbox("Gespeicherte Signale anzeigen (Analyse-Nr.)", verlauf["id"].tolist())
    if nummer is not None:
        st.dataframe(gespeicherte_signale_laden(int(nummer)), hide_index=True)
    with b:
        bestaetigt = st.checkbox("Löschen bestätigen")
        if st.button("Alle gespeicherten Analysen löschen", disabled=not bestaetigt, icon=":material/delete:"):
            analysen_loeschen()
            st.rerun()


def _tab_datenqualitaet(q: DatenQualitaet, paket: DatenPaket, erg: AnalyseErgebnis) -> None:
    for warnung in q.warnungen:
        st.warning(warnung, icon=":material/warning:")
    for hinweis in paket.hinweise + q.hinweise + erg.hinweise:
        st.info(hinweis, icon=":material/info:")
    intraday = erg.intervall.intraday
    uebersicht = pd.DataFrame([
        ("Datenquelle / Symbol", f"{paket.quelle} / {paket.symbol}"),
        ("Geladene Zeilen (inkl. Vorlauf)", fmt_zahl(q.zeilen_roh, 0)),
        ("Zeilen nach Bereinigung", fmt_zahl(q.zeilen_bereinigt, 0)),
        ("Kerzen im Anzeigezeitraum", fmt_zahl(len(erg.d), 0)),
        ("Erste / letzte Kerze", f"{fmt_datum(q.erste_kerze, intraday)} / {fmt_datum(q.letzte_kerze, intraday)}"),
        ("Fehlende Werte (roh)", ", ".join(f"{k}: {v}" for k, v in q.fehlende_werte.items()) or "keine"),
        ("Korrigierte OHLC-Kerzen", fmt_zahl(q.korrigierte_ohlc, 0)),
        ("Größere Lücken", fmt_zahl(len(q.luecken), 0)),
        ("Verdächtige Kurssprünge", fmt_zahl(len(q.kurs_spruenge), 0)),
        ("Volumen verfügbar", "ja" if q.volumen_verfuegbar else f"nein ({fmt_pct(q.null_volumen_anteil, 0, False)} ohne Volumen)"),
        ("Letzte Kerze abgeschlossen", "nein (vorläufig)" if erg.letzte_kerze_vorlaeufig else "ja"),
    ], columns=["Prüfung", "Ergebnis"])
    st.dataframe(uebersicht, hide_index=True, height=_tabellenhoehe(len(uebersicht)))
    if not q.kurs_spruenge.empty:
        st.markdown("**Verdächtige Kurssprünge**")
        st.dataframe(q.kurs_spruenge.style.format({"Datum": lambda t: fmt_datum(t), "Vortag": fmt_zahl,
                                                   "Schluss": fmt_zahl, "Veränderung": fmt_pct}), hide_index=True)
    if q.luecken:
        st.markdown("**Lücken in der Zeitreihe**")
        st.dataframe(pd.DataFrame(q.luecken).style.format({"von": lambda t: fmt_datum(t, intraday),
                                                           "bis": lambda t: fmt_datum(t, intraday)}), hide_index=True)
    st.caption(DATENHINWEIS)


def streamlit_app_starten() -> None:
    """Startet die Streamlit-Oberfläche."""
    if st is None:
        raise RuntimeError("Streamlit ist nicht installiert: pip install -r requirements.txt")
    st.set_page_config(page_title=APP_NAME, page_icon="📈", layout="wide", initial_sidebar_state="expanded")
    st.markdown(_CSS, unsafe_allow_html=True)
    _startwerte_setzen()
    if "ansicht" not in st.session_state:
        st.session_state["ansicht"] = (ANSICHT_BUDGET if str(st.query_params.get("ansicht", "")).lower() == "budget"
                                       else ANSICHT_EINZEL)
    ansicht = st.sidebar.radio("Ansicht", [ANSICHT_EINZEL, ANSICHT_BUDGET], key="ansicht")
    if ansicht == ANSICHT_BUDGET:
        budget_ansicht_starten()
        return
    e = _seitenleiste()

    st.title(APP_NAME)
    st.caption("Aktien und Indizes der Borsa İstanbul · Indikatoren, Signalbewertung, Risikoanalyse und Backtest")
    st.warning(HAFTUNGSAUSSCHLUSS, icon=":material/gavel:")

    fehler = e["sp"].pruefen()
    if fehler:
        for text in fehler:
            st.error(text)
        st.stop()

    info = INTERVALLE[e["intervall"]]
    try:
        with st.spinner("Lade Kursdaten …"):
            paket = _daten_laden_gecacht(e["ticker"], e["zeitraum"], e["intervall"], e["quelle"], e["suffix"],
                                         e["unveraendert"], e["dividende"], e["csv_inhalt"])
    except DatenFehler as fehler_daten:
        _fehler_anzeigen(fehler_daten)
        st.stop()
    except Exception as exc:  # unerwartete Fehler verständlich anzeigen
        st.error(f"Unerwarteter Fehler beim Laden der Daten: {type(exc).__name__}: {exc}")
        st.stop()

    live = paket.quelle in LIVE_QUELLEN
    daten, qualitaet = daten_validieren(paket.daten, info, live=live)
    if daten.empty:
        st.error("Nach der Datenprüfung sind keine gültigen Kurse vorhanden.")
        st.stop()

    waehrung = paket.waehrung
    if e["waehrung"] != "TRY":
        if waehrung != "TRY":
            st.info(f"Der Wert notiert in {waehrung}; eine Umrechnung ist nur für TRY-Kurse vorgesehen.")
        elif info.intraday:
            st.info("Die Währungsumrechnung ist nur für Tages- und Wochendaten verfügbar – Anzeige in TRY.")
        else:
            try:
                with st.spinner("Lade Wechselkurse …"):
                    fx = _wechselkurse_gecacht(e["waehrung"], daten.index.min(), daten.index.max() + pd.Timedelta(days=1))
                daten = waehrung_umrechnen(daten, fx)
                waehrung = e["waehrung"]
            except DatenFehler as fehler_fx:
                st.warning(f"Wechselkurse nicht verfügbar ({fehler_fx.meldung}) – Anzeige in TRY.")

    daten_iv = auf_intervall_bringen(daten, info)
    vorlaeufig = live and letzte_kerze_unvollstaendig(daten_iv.index, info)
    try:
        erg = analyse_durchfuehren(daten_iv, paket.anzeige_start, info, e["sp"], e["rp"], e["bp"],
                                   qualitaet.volumen_verfuegbar, vorlaeufig)
    except Exception as exc:
        st.error(f"Die Analyse konnte nicht durchgeführt werden: {type(exc).__name__}: {exc}")
        st.stop()

    uebersicht = kursuebersicht_berechnen(daten, info)
    if paket.quelle == QUELLE_DEMO:
        st.error("DEMO-MODUS: Die angezeigten Kurse sind synthetisch erzeugt und keine echten Marktdaten.",
                 icon=":material/science:")
    for hinweis in paket.hinweise:
        if not hinweis.startswith("DEMO"):
            st.info(hinweis, icon=":material/info:")
    if qualitaet.warnungen:
        st.warning("**Datenqualität:** " + " ".join(qualitaet.warnungen[:2]) + " (Details im Reiter „Datenqualität“)",
                   icon=":material/warning:")

    einheit = kurs_einheit(paket, waehrung)
    _kopfbereich(paket, uebersicht, waehrung, info, e["profil"])
    kurzfazit = kurzfazit_erstellen(erg, paket.name, paket.symbol, einheit, uebersicht)
    _gesamtsignal_bereich(erg, kurzfazit, einheit)
    bericht = bericht_erstellen(erg, paket.name, paket.symbol, paket.quelle, paket.zeitraum, paket.intervall,
                                waehrung, uebersicht, qualitaet, einheit)

    reiter = st.tabs(["Chart", "Indikatoren", "Signalbegründung", "Risikoanalyse", "Backtest", "Signalhistorie",
                      "Bericht & Export", "Datenqualität"])
    with reiter[0]:
        fig = _tab_chart(erg, e, f"{paket.name} ({paket.symbol}) · {paket.intervall}", einheit)
    with reiter[1]:
        _tab_indikatoren(erg)
    with reiter[2]:
        _tab_begruendung(erg)
    with reiter[3]:
        _tab_risiko(erg, waehrung, einheit)
    with reiter[4]:
        _tab_backtest(erg, waehrung)
    with reiter[5]:
        _tab_signalhistorie(erg)
    with reiter[6]:
        _tab_export(erg, paket, fig, bericht, waehrung)
        st.divider()
        _tab_datenbank(erg, paket, bericht, waehrung)
    with reiter[7]:
        _tab_datenqualitaet(qualitaet, paket, erg)

    st.divider()
    st.caption(f"{HAFTUNGSAUSSCHLUSS} Kursdaten können verzögert, unvollständig oder fehlerhaft sein.")


# =============================================================================
# 14b) Budgetplanung und Aktienauswahl (regelbasierte Beispielanalyse)
# =============================================================================

BUDGET_WARNHINWEIS = (
    "Diese Analyse dient ausschließlich zu Informations- und Bildungszwecken und stellt keine Anlageberatung, "
    "Kaufempfehlung oder Finanzberatung dar. Technische Signale können falsch sein. Investiere nur Geld, dessen "
    "Verlust du dir leisten kannst. Vor einer tatsächlichen Investition müssen aktuelle Marktdaten, "
    "Unternehmensinformationen, Gebühren, Steuern und persönliche Risikobereitschaft geprüft werden."
)
BUDGET_TITEL = "Regelbasierte Beispielanalyse – keine persönliche Anlageempfehlung."
TOP_FORMULIERUNG = "Nach den ausgewählten technischen Kriterien aktuell das stärkste positive Signal."

# Signalstufen der Aktienauswahl (bewusst neutrale Formulierungen)
SCREEN_STARK_KAUF = "Starkes Kaufsignal"
SCREEN_KAUF = "Kaufsignal"
SCREEN_BEOBACHTEN = "Beobachten"
SCREEN_VERKAUF = "Verkaufssignal"
SCREEN_STARK_VERKAUF = "Starkes Verkaufssignal"
SCREEN_STIL = {SCREEN_STARK_KAUF: "stark_pos", SCREEN_KAUF: "pos", SCREEN_BEOBACHTEN: "neutral",
               SCREEN_VERKAUF: "neg", SCREEN_STARK_VERKAUF: "stark_neg"}
SIGNAL_STIL.update(SCREEN_STIL)   # Farben auch in Tabellen verwenden

# Standardliste für die Auswahl (Aktien, keine Indizes) – in der Oberfläche änderbar
STANDARD_AKTIENLISTE = ["THYAO", "GARAN", "AKBNK", "ISCTR", "YKBNK", "ASELS", "BIMAS", "KCHOL", "SAHOL",
                        "EREGL", "TUPRS", "SISE", "FROTO", "TOASO", "PGSUS", "TCELL", "ENKAI", "ARCLK",
                        "MGROS", "TAVHL"]

RISIKOPROFILE: dict[str, dict[str, float]] = {
    # max_pos/min_anteil/max_anteil/reserve in %, Verluste in % des Budgets, max_vola = Volatilität p. a. in %
    "konservativ": {"max_positionen": 4, "min_anteil": 10.0, "max_anteil": 30.0, "reserve": 30.0,
                    "verlust_position": 0.5, "verlust_portfolio": 3.0, "max_vola": 50.0},
    "mittel": {"max_positionen": 6, "min_anteil": 8.0, "max_anteil": 25.0, "reserve": 15.0,
               "verlust_position": 1.0, "verlust_portfolio": 6.0, "max_vola": 65.0},
    "hoch": {"max_positionen": 8, "min_anteil": 5.0, "max_anteil": 30.0, "reserve": 5.0,
             "verlust_position": 2.0, "verlust_portfolio": 12.0, "max_vola": 90.0},
}
RISIKO_LABEL = {"konservativ": "konservatives Risiko", "mittel": "mittleres Risiko", "hoch": "hohes Risiko"}

HORIZONTE: dict[str, dict[str, Any]] = {
    # Gewichte der Bewertungskomponenten je Anlagehorizont (transparent und änderbar)
    "kurzfristig": {"tage": 20, "atr_multiplikator": 1.5, "crv": 1.5,
                    "gewichte": {"basis": 1.0, "trend": 1.0, "ema": 2.0, "sma50": 1.0, "sma_lang": 0.5,
                                 "obv": 1.0, "vola": 1.0}},
    "mittelfristig": {"tage": 60, "atr_multiplikator": 2.0, "crv": 2.0,
                      "gewichte": {"basis": 1.0, "trend": 1.5, "ema": 1.0, "sma50": 1.0, "sma_lang": 1.0,
                                   "obv": 1.0, "vola": 1.0}},
    "langfristig": {"tage": 250, "atr_multiplikator": 3.0, "crv": 2.5,
                    "gewichte": {"basis": 1.0, "trend": 2.0, "ema": 0.5, "sma50": 0.5, "sma_lang": 2.0,
                                 "obv": 0.5, "vola": 1.5}},
}
KOMPONENTEN_NAMEN = {
    "basis": "Punktesystem (RSI, MACD, Golden Cross, SMA 200, Volumen, ADX)",
    "trend": "Trendstärke (Trend + ADX)", "ema": "EMA 12 vs. EMA 26", "sma50": "Kurs vs. SMA 50",
    "sma_lang": "SMA 50 vs. SMA 200", "obv": "Volumenfluss (OBV vs. Ø 20)", "vola": "Volatilität",
    "zonen": "Unterstützung/Widerstand", "stop": "Abstand zum Stop-Loss", "liquiditaet": "Liquidität",
}


@dataclass
class BudgetParameter:
    """Einstellungen der Budgetplanung (Standardwerte aus Risikoprofil und Anlagehorizont)."""

    budget: float = 100_000.0
    risikoprofil: str = "mittel"
    horizont: str = "mittelfristig"
    max_positionen: int = 6
    min_anteil_pct: float = 8.0
    max_anteil_pct: float = 25.0
    reserve_pct: float = 15.0
    reserve_min_tl: float = 0.0
    verlust_position_pct: float = 1.0
    verlust_portfolio_pct: float = 6.0
    atr_multiplikator: float = 2.0
    crv: float = 2.0
    gebuehren_pct: float = 0.10
    slippage_pct: float = 0.05
    steuer_pct: float = 0.0
    max_vola_pct: float = 65.0
    min_liquiditaet_tl: float = 20_000_000.0

    @classmethod
    def aus_profil(cls, budget: float, risikoprofil: str = "mittel", horizont: str = "mittelfristig",
                   **abweichungen: Any) -> BudgetParameter:
        p, h = RISIKOPROFILE[risikoprofil], HORIZONTE[horizont]
        werte: dict[str, Any] = dict(
            budget=budget, risikoprofil=risikoprofil, horizont=horizont, max_positionen=int(p["max_positionen"]),
            min_anteil_pct=p["min_anteil"], max_anteil_pct=p["max_anteil"], reserve_pct=p["reserve"],
            verlust_position_pct=p["verlust_position"], verlust_portfolio_pct=p["verlust_portfolio"],
            atr_multiplikator=h["atr_multiplikator"], crv=h["crv"], max_vola_pct=p["max_vola"])
        werte.update(abweichungen)
        return cls(**werte)

    def pruefen(self) -> list[str]:
        fehler = []
        if not (_ist_zahl(self.budget) and self.budget > 0):
            fehler.append("Bitte einen positiven Betrag in TL eingeben.")
        if not (0 < self.min_anteil_pct <= self.max_anteil_pct <= 50):
            fehler.append("Anteile je Aktie: 0 < Mindestanteil ≤ Höchstanteil ≤ 50 % (nie alles in eine Aktie).")
        if not (0 <= self.reserve_pct < 100):
            fehler.append("Die Liquiditätsreserve muss zwischen 0 und 100 % liegen.")
        if self.verlust_position_pct <= 0 or self.verlust_portfolio_pct <= 0:
            fehler.append("Die Verlustgrenzen müssen größer als 0 sein.")
        if self.max_positionen < 2:
            fehler.append("Mindestens zwei Positionen, damit nicht alles in eine Aktie fließt.")
        return fehler


def fmt_tl(betrag: Any, nachkomma: int = 2) -> str:
    """Betrag in türkischer Lira, z. B. 100.000,00 ₺."""
    return "–" if not _ist_zahl(betrag) else f"{fmt_zahl(betrag, nachkomma)} ₺"


def markt_status(jetzt: pd.Timestamp | None = None) -> dict[str, Any]:
    """Öffnungsstatus der Borsa İstanbul nach Uhrzeit (Feiertage werden nicht berücksichtigt)."""
    jetzt = jetzt if jetzt is not None else _jetzt_istanbul()
    minuten = jetzt.hour * 60 + jetzt.minute
    if jetzt.weekday() >= 5:
        text, offen = "geschlossen (Wochenende)", False
    elif 9 * 60 + 40 <= minuten < 10 * 60:
        text, offen = "Eröffnungsauktion (09:40–10:00 Uhr)", True
    elif 10 * 60 <= minuten < 18 * 60:
        text, offen = "geöffnet – fortlaufender Handel (10:00–18:00 Uhr)", True
    elif 18 * 60 <= minuten < BIST_SITZUNGSENDE_MIN:
        text, offen = "Schlussauktion (18:00–18:10 Uhr)", True
    else:
        text, offen = "geschlossen (außerhalb der Handelszeit)", False
    return {"offen": offen, "text": text, "zeit": jetzt,
            "hinweis": "Nach Uhrzeit Istanbul bestimmt; Feiertage und Handelsunterbrechungen sind nicht berücksichtigt."}


def erwarteter_handelstag(jetzt: pd.Timestamp | None = None) -> pd.Timestamp:
    """Letzter Handelstag, für den Tageskurse vorliegen sollten (ohne Feiertagskalender)."""
    jetzt = jetzt if jetzt is not None else _jetzt_istanbul()
    tag = jetzt.normalize()
    if jetzt.weekday() < 5 and jetzt.hour * 60 + jetzt.minute >= 10 * 60:
        return tag
    tag -= pd.Timedelta(days=1)
    while tag.weekday() >= 5:
        tag -= pd.Timedelta(days=1)
    return tag


@dataclass
class AktienBewertung:
    """Bewertung einer Aktie für die Auswahl (alle Werte zum letzten verfügbaren Kurs)."""

    ticker: str
    symbol: str = ""
    name: str = ""
    ok: bool = False
    ausschlussgrund: str | None = None
    kurs: float = float("nan")
    veraenderung_pct: float = float("nan")
    datum: pd.Timestamp | None = None
    aktuell: bool = False
    vorlaeufig: bool = False
    punkte: float = 0.0
    max_punkte: float = 1.0
    score_pct: float = 0.0
    signal: str = SCREEN_BEOBACHTEN
    signal_vortag: str = SCREEN_BEOBACHTEN
    neu: bool = False
    komponenten: dict[str, float] = field(default_factory=dict)
    gruende: list[str] = field(default_factory=list)
    risiken: list[str] = field(default_factory=list)
    unterstuetzung: Zone | None = None
    widerstand: Zone | None = None
    atr: float = float("nan")
    stop: float = float("nan")
    ziel: float = float("nan")
    crv: float = float("nan")
    crv_bis_widerstand: float = float("nan")
    vola_pct: float = float("nan")
    liquiditaet_tl: float = float("nan")
    risiko_stufe: str = "–"
    ereignissignal: str = HALTEN


def _screen_signal(anteil: float) -> str:
    if anteil >= 0.5:
        return SCREEN_STARK_KAUF
    if anteil >= 0.25:
        return SCREEN_KAUF
    if anteil <= -0.5:
        return SCREEN_STARK_VERKAUF
    if anteil <= -0.25:
        return SCREEN_VERKAUF
    return SCREEN_BEOBACHTEN


def _komponenten_reihen(d: pd.DataFrame, sp: StrategieParameter, max_vola: float) -> dict[str, pd.Series]:
    """Vektorisierte Bewertungskomponenten je Kerze (nur vergangene Daten)."""
    adx_stark = d["ADX_14"] >= sp.adx_schwelle
    trend = d["Trend"].astype(float) * np.where(adx_stark, 2.0, 1.0)
    vola = np.log(d["Close"]).diff().rolling(20, min_periods=15).std() * math.sqrt(HANDELSTAGE_PRO_JAHR) * 100
    vola_punkte = np.select([vola > max_vola, vola > 0.8 * max_vola], [-2.0, -1.0], 0.0)

    def vorzeichen(a: pd.Series, b: pd.Series) -> pd.Series:
        return pd.Series(np.select([a > b, a < b], [1.0, -1.0], 0.0), index=d.index)

    return {
        "basis": d["Punkte"].astype(float),
        "trend": pd.Series(trend, index=d.index).fillna(0.0),
        "ema": vorzeichen(d["EMA_12"], d["EMA_26"]),
        "sma50": vorzeichen(d["Close"], d["SMA_50"]),
        "sma_lang": vorzeichen(d["SMA_50"], d["SMA_200"]),
        "obv": vorzeichen(d["OBV"], d["OBV_SMA_20"]),
        "vola": pd.Series(vola_punkte, index=d.index),
        "_vola_pct": vola,
    }


def _max_positive_punkte(gewichte: dict[str, float], sp: StrategieParameter) -> float:
    """Höchste erreichbare positive Punktzahl (für die Prozentangabe)."""
    return (gewichte["basis"] * max(sp.gewichte.maximum(), 1) + gewichte["trend"] * 2 + gewichte["ema"]
            + gewichte["sma50"] + gewichte["sma_lang"] + gewichte["obv"] + 1.0)   # +1 für Zonen


def aktie_bewerten(ticker: str, daten: pd.DataFrame, bp: BudgetParameter, sp: StrategieParameter | None = None,
                   name: str = "", symbol: str = "", live: bool = True,
                   jetzt: pd.Timestamp | None = None) -> AktienBewertung:
    """Bewertet eine Aktie nach transparenten technischen Kriterien.

    Komponenten: Punktesystem (RSI, MACD, Kreuzungen, SMA 200, Volumen, ADX), Trendstärke, EMA-/SMA-Verhältnisse,
    Volumenfluss, Volatilität sowie – nur für die letzte Kerze – Unterstützung/Widerstand, Stop-Abstand und
    Liquidität. Gewichte hängen vom Anlagehorizont ab (``HORIZONTE``).
    """
    sp = sp or StrategieParameter()
    jetzt = jetzt if jetzt is not None else _jetzt_istanbul()
    b = AktienBewertung(ticker=ticker, symbol=symbol or ticker,
                        name=name or BIST_FAVORITEN.get(ticker) or BIST_ALLE_AKTIEN.get(ticker, ticker))
    info = INTERVALLE["Täglich"]
    bereinigt, qualitaet = daten_validieren(daten, info, jetzt=jetzt, live=live)
    if len(bereinigt) < 220:
        b.ausschlussgrund = f"Zu wenige Kursdaten ({len(bereinigt)} Handelstage, mindestens 220 nötig)."
        return b
    b.datum = bereinigt.index[-1]
    b.aktuell = (not live) or b.datum >= erwarteter_handelstag(jetzt)
    if live and len(pd.bdate_range(b.datum + pd.Timedelta(days=1), jetzt.normalize())) > 5:
        b.ausschlussgrund = f"Veraltete Daten (letzte Kerze {fmt_datum(b.datum)})."
        return b
    if not qualitaet.volumen_verfuegbar:
        b.ausschlussgrund = "Keine verlässlichen Volumendaten."
        return b
    juengste_spruenge = qualitaet.kurs_spruenge
    if not juengste_spruenge.empty and (juengste_spruenge["Datum"] >= bereinigt.index[-60]).any():
        b.ausschlussgrund = "Unplausibler Kurssprung in den letzten 60 Tagen (mögliche Kapitalmaßnahme/Datenfehler)."
        return b

    # Schnelle Vorprüfung (vor der aufwendigen Indikatorberechnung): Liquidität und Volatilität
    liquiditaet = float((bereinigt["Close"] * bereinigt["Volume"]).tail(20).mean())
    if _ist_zahl(liquiditaet) and liquiditaet < bp.min_liquiditaet_tl / 5:
        b.ausschlussgrund = f"Sehr geringe Liquidität (Ø Umsatz {fmt_volumen(liquiditaet)} TL/Tag)."
        return b
    vola_vorab = float(np.log(bereinigt["Close"]).diff().tail(20).std() * math.sqrt(HANDELSTAGE_PRO_JAHR) * 100)
    if _ist_zahl(vola_vorab) and vola_vorab > bp.max_vola_pct:
        b.ausschlussgrund = (f"Volatilität {fmt_zahl(vola_vorab, 0)} % p. a. über der Grenze des Risikoprofils "
                             f"({fmt_zahl(bp.max_vola_pct, 0)} %).")
        return b

    d = indikatoren_berechnen(bereinigt)
    d, _ = trend_analyse(d, sp)
    d = signalpunkte_berechnen(d, sp, True)
    d, _ = kauf_und_verkaufssignale_ermitteln(d, sp, True)
    z = d.iloc[-1]
    b.kurs = float(z["Close"])
    b.veraenderung_pct = float(d["Close"].iloc[-1] / d["Close"].iloc[-2] - 1)
    b.vorlaeufig = live and letzte_kerze_unvollstaendig(d.index, info, jetzt)
    b.ereignissignal = aktueller_signalstatus(d, sp)["status"]
    b.atr = float(z["ATR_14"])
    b.liquiditaet_tl = float((d["Close"] * d["Volume"]).tail(20).mean())

    gewichte = HORIZONTE[bp.horizont]["gewichte"]
    reihen = _komponenten_reihen(d, sp, bp.max_vola_pct)
    b.vola_pct = float(reihen.pop("_vola_pct").iloc[-1])
    if _ist_zahl(b.vola_pct) and b.vola_pct > bp.max_vola_pct:
        b.ausschlussgrund = (f"Volatilität {fmt_zahl(b.vola_pct, 0)} % p. a. über der Grenze des Risikoprofils "
                             f"({fmt_zahl(bp.max_vola_pct, 0)} %).")
        return b
    if _ist_zahl(b.liquiditaet_tl) and b.liquiditaet_tl < bp.min_liquiditaet_tl / 5:
        b.ausschlussgrund = f"Sehr geringe Liquidität (Ø Umsatz {fmt_volumen(b.liquiditaet_tl)} TL/Tag)."
        return b
    vektor = sum(gewichte[k] * reihen[k] for k in gewichte)
    b.max_punkte = _max_positive_punkte(gewichte, sp)

    # Komponenten der letzten Kerze
    komponenten = {k: float(gewichte[k] * reihen[k].iloc[-1]) for k in gewichte}
    b.stop = b.kurs - bp.atr_multiplikator * b.atr
    b.ziel = b.kurs + bp.crv * (b.kurs - b.stop)
    b.crv = bp.crv
    zonen = unterstuetzung_widerstand_ermitteln(d.tail(250))
    unter = [zz for zz in zonen if zz.art == "Unterstützung"]
    ueber = [zz for zz in zonen if zz.art == "Widerstand"]
    b.unterstuetzung = max(unter, key=lambda zz: zz.oben) if unter else None
    b.widerstand = min(ueber, key=lambda zz: zz.unten) if ueber else None
    risiko = b.kurs - b.stop
    zonen_punkte = 0.0
    if b.widerstand is not None and risiko > 0:
        b.crv_bis_widerstand = (b.widerstand.unten - b.kurs) / risiko
        if b.crv_bis_widerstand < 1:
            zonen_punkte -= 1.0
    if b.unterstuetzung is not None and b.kurs - b.unterstuetzung.oben <= b.atr:
        zonen_punkte += 1.0
    komponenten["zonen"] = zonen_punkte
    stop_abstand = risiko / b.kurs if b.kurs else float("nan")
    komponenten["stop"] = -1.0 if stop_abstand > 0.12 else 0.0
    komponenten["liquiditaet"] = -1.0 if b.liquiditaet_tl < bp.min_liquiditaet_tl else 0.0
    b.komponenten = komponenten
    b.punkte = float(sum(komponenten.values()))
    b.score_pct = b.punkte / b.max_punkte
    b.signal = _screen_signal(b.score_pct)
    vektor_vortag = float(vektor.iloc[-2]) / b.max_punkte
    b.signal_vortag = _screen_signal(vektor_vortag)
    b.neu = b.signal != SCREEN_BEOBACHTEN and _screen_signal(float(vektor.iloc[-1]) / b.max_punkte) != b.signal_vortag
    b.ok = True

    # Verständliche Begründungen (größte Beiträge zuerst)
    texte = {
        "basis": f"Punktesystem {punkte_text(z['Punkte'])} (RSI {fmt_zahl(z['RSI_14'], 0)}, MACD "
                 f"{'über' if z['MACD'] > z['MACD_Signal'] else 'unter'} Signallinie)",
        "trend": f"Trend {TREND_TEXT[int(z['Trend'])]}, ADX {fmt_zahl(z['ADX_14'], 0)}",
        "ema": f"EMA 12 {'über' if z['EMA_12'] > z['EMA_26'] else 'unter'} EMA 26",
        "sma50": f"Kurs {'über' if z['Close'] > z['SMA_50'] else 'unter'} SMA 50",
        "sma_lang": f"SMA 50 {'über' if z['SMA_50'] > z['SMA_200'] else 'unter'} SMA 200",
        "obv": f"OBV {'über' if z['OBV'] > z['OBV_SMA_20'] else 'unter'} Ø 20 (Volumenfluss)",
        "vola": f"Volatilität {fmt_zahl(b.vola_pct, 0)} % p. a.",
        "zonen": "Kurs nahe Unterstützung" if zonen_punkte > 0 else
                 f"Widerstand nahe (CRV bis Widerstand {fmt_zahl(b.crv_bis_widerstand, 1)})",
        "stop": f"Stop-Abstand {fmt_pct(stop_abstand, 1, False)} (groß)",
        "liquiditaet": f"Geringe Liquidität (Ø {fmt_volumen(b.liquiditaet_tl)} TL/Tag)",
    }
    for schluessel, wert in sorted(komponenten.items(), key=lambda kv: -abs(kv[1])):
        if wert > 0:
            b.gruende.append(f"{texte[schluessel]} ({fmt_zahl(wert, 1, True)})")
        elif wert < 0:
            b.risiken.append(f"{texte[schluessel]} ({fmt_zahl(wert, 1, True)})")
    if b.vorlaeufig:
        b.risiken.append("Letzte Kerze noch nicht abgeschlossen – Werte vorläufig.")
    if not b.aktuell:
        b.risiken.append(f"Kursdaten nicht vom aktuellen Handelstag (Stand {fmt_datum(b.datum)}).")
    b.risiko_stufe = ("hoch" if b.vola_pct >= 55 or stop_abstand > 0.12 else
                      "mittel" if b.vola_pct >= 35 or stop_abstand > 0.07 else "niedrig")
    return b


def aktien_screenen(ticker_liste: Sequence[str], lade_funktion: Any, bp: BudgetParameter,
                    sp: StrategieParameter | None = None, live: bool = True,
                    jetzt: pd.Timestamp | None = None, fortschritt: Any = None) -> list[AktienBewertung]:
    """Lädt und bewertet alle Aktien der Liste; sortiert nach Score (ausgeschlossene am Ende)."""
    ergebnisse: list[AktienBewertung] = []
    for nummer, eingabe in enumerate(ticker_liste, start=1):
        if fortschritt is not None:
            fortschritt(nummer / max(len(ticker_liste), 1), eingabe)
        try:
            basis = ticker_normalisieren(eingabe).basis
        except DatenFehler as fehler:
            ergebnisse.append(AktienBewertung(ticker=str(eingabe), ausschlussgrund=fehler.meldung))
            continue
        try:
            paket = lade_funktion(basis)
            if paket.ist_index:
                ergebnisse.append(AktienBewertung(ticker=basis, ausschlussgrund="Index – nicht direkt investierbar."))
                continue
            ergebnisse.append(aktie_bewerten(basis, paket.daten, bp, sp, name=paket.name, symbol=paket.symbol,
                                             live=live, jetzt=jetzt))
        except DatenFehler as fehler:
            ergebnisse.append(AktienBewertung(ticker=basis, ausschlussgrund=fehler.meldung))
        except Exception as exc:  # einzelne Fehler dürfen die Auswertung nicht abbrechen
            ergebnisse.append(AktienBewertung(ticker=basis, ausschlussgrund=f"Fehler: {type(exc).__name__}: {exc}"))
    return sorted(ergebnisse, key=lambda x: (not x.ok, -x.score_pct if x.ok else 0))


@dataclass
class PortfolioErgebnis:
    positionen: pd.DataFrame
    budget: float
    reserve: float
    investiert: float
    gebuehren: float
    liquiditaet: float
    max_verlust: float
    hinweise: list[str] = field(default_factory=list)


def portfolio_aufteilen(bewertungen: Sequence[AktienBewertung], bp: BudgetParameter) -> PortfolioErgebnis:
    """Beispielhafte, risikobasierte Aufteilung des Budgets.

    Stückzahl = min(Risikobudget je Position / Positionsrisiko, Zielbetrag / Kurs inkl. Gebühren);
    Positionsrisiko = Einstiegskurs − Stop-Loss (+ Gebühren und Slippage). Anschließend werden
    Höchstanteil, Mindestanteil, Gesamtrisiko des Portfolios und die Liquiditätsreserve eingehalten.
    Es werden nur ganze Aktien verwendet (BIST: Handel in ganzen Stück).
    """
    gebuehr, slip = bp.gebuehren_pct / 100, bp.slippage_pct / 100
    reserve = min(max(bp.budget * bp.reserve_pct / 100, bp.reserve_min_tl), bp.budget)
    investierbar = bp.budget - reserve
    hinweise: list[str] = []
    kandidaten = [b for b in bewertungen if b.ok and b.signal in (SCREEN_STARK_KAUF, SCREEN_KAUF)
                  and _ist_zahl(b.atr) and b.atr > 0][: bp.max_positionen]
    spalten = ["Aktie", "Ticker", "Signal", "Score", "Betrag (TL)", "Anteil", "Stück", "Kurs (verwendet)",
               "Restbetrag (TL)", "Stop-Loss", "Kursziel", "Max. Verlust bis Stop (TL)", "Gebühren (TL)",
               "Gründe", "Risiken"]
    if not kandidaten:
        hinweise.append("Keine Aktie erfüllt aktuell die Kriterien für ein Kaufsignal – das Budget bleibt vollständig "
                        "Liquidität.")
        return PortfolioErgebnis(pd.DataFrame(columns=spalten), bp.budget, reserve, 0.0, 0.0, bp.budget, 0.0, hinweise)
    if len(kandidaten) < 2:
        hinweise.append("Nur eine Aktie mit positivem Signal – der Höchstanteil begrenzt die Position, der Rest bleibt "
                        "Liquidität.")

    # Zielanteile nach Score gewichten und auf Mindest-/Höchstanteil begrenzen
    gewichte = np.array([max(b.score_pct, 0.01) for b in kandidaten])
    anteile = gewichte / gewichte.sum() * investierbar / bp.budget * 100
    anteile = np.clip(anteile, bp.min_anteil_pct, bp.max_anteil_pct)
    if anteile.sum() > investierbar / bp.budget * 100:
        anteile *= (investierbar / bp.budget * 100) / anteile.sum()

    zeilen = []
    for b, anteil in zip(kandidaten, anteile, strict=True):
        einstieg = b.kurs * (1 + slip)
        stop = einstieg - bp.atr_multiplikator * b.atr
        if stop <= 0:
            hinweise.append(f"{b.ticker}: Stop-Loss wäre ≤ 0 – ausgelassen.")
            continue
        ziel = einstieg + bp.crv * (einstieg - stop)
        risiko_je_aktie = (einstieg - stop) + einstieg * gebuehr + stop * (gebuehr + slip)
        stueck_risiko = math.floor(bp.budget * bp.verlust_position_pct / 100 / risiko_je_aktie)
        zielbetrag = bp.budget * anteil / 100
        stueck_kapital = math.floor(zielbetrag / (einstieg * (1 + gebuehr)))
        stueck = max(0, min(stueck_risiko, stueck_kapital))
        begrenzung = "Risiko" if stueck_risiko < stueck_kapital else "Betrag"
        zeilen.append({"b": b, "anteil_ziel": anteil, "einstieg": einstieg, "stop": stop, "ziel": ziel,
                       "risiko": risiko_je_aktie, "stueck": stueck, "zielbetrag": zielbetrag, "grenze": begrenzung})

    # Gesamtrisiko des Portfolios begrenzen
    limit = bp.budget * bp.verlust_portfolio_pct / 100
    gesamt = sum(z["stueck"] * z["risiko"] for z in zeilen)
    if gesamt > limit > 0:
        faktor = limit / gesamt
        for z in zeilen:
            z["stueck"] = math.floor(z["stueck"] * faktor)
        hinweise.append(f"Stückzahlen wurden gekürzt, damit der maximale Gesamtverlust "
                        f"{fmt_pct(bp.verlust_portfolio_pct / 100, 1, False)} des Budgets nicht übersteigt.")

    ergebnis_zeilen, investiert, gebuehren_summe, max_verlust = [], 0.0, 0.0, 0.0
    for z in zeilen:
        b = z["b"]
        if z["stueck"] <= 0:
            hinweise.append(f"{b.ticker}: Budget bzw. Risikogrenze reicht nicht für eine ganze Aktie.")
            continue
        betrag = z["stueck"] * z["einstieg"]
        kosten = betrag * gebuehr
        if betrag < bp.budget * bp.min_anteil_pct / 100 * 0.5:
            hinweise.append(f"{b.ticker}: Position durch das Risikolimit klein ({fmt_tl(betrag, 0)}).")
        verlust = z["stueck"] * z["risiko"]
        investiert += betrag
        gebuehren_summe += kosten
        max_verlust += verlust
        ergebnis_zeilen.append({
            "Aktie": b.name, "Ticker": b.ticker, "Signal": b.signal, "Score": round(b.score_pct * 100, 1),
            "Betrag (TL)": betrag, "Anteil": betrag / bp.budget, "Stück": int(z["stueck"]),
            "Kurs (verwendet)": z["einstieg"], "Restbetrag (TL)": z["zielbetrag"] - betrag - kosten,
            "Stop-Loss": z["stop"], "Kursziel": z["ziel"], "Max. Verlust bis Stop (TL)": verlust,
            "Gebühren (TL)": kosten, "Gründe": "; ".join(b.gruende[:3]) or "–",
            "Risiken": "; ".join(b.risiken[:3] + [f"Stückzahl begrenzt durch {z['grenze']}"]),
        })
    positionen = pd.DataFrame(ergebnis_zeilen, columns=spalten)
    if len(positionen) == 1:
        hinweise.append("Nur eine Position möglich – bewusst keine Investition des gesamten Betrags in diese Aktie.")
    liquiditaet = bp.budget - investiert - gebuehren_summe
    return PortfolioErgebnis(positionen, bp.budget, reserve, investiert, gebuehren_summe, liquiditaet, max_verlust,
                             hinweise)


def szenarien_berechnen(portfolio: PortfolioErgebnis, bewertungen: Sequence[AktienBewertung],
                        bp: BudgetParameter) -> pd.DataFrame:
    """Drei beispielhafte Szenarien über den Anlagehorizont (keine Prognose)."""
    tage = HORIZONTE[bp.horizont]["tage"]
    gebuehr, slip, steuer = bp.gebuehren_pct / 100, bp.slippage_pct / 100, bp.steuer_pct / 100
    nach_ticker = {b.ticker: b for b in bewertungen}
    beschreibung = {
        "Vorsichtiges Szenario": f"Jede Aktie fällt um eine Standardabweichung über {tage} Handelstage; "
                                 "unterhalb des Stop-Loss wird zum Stop verkauft.",
        "Neutrales Szenario": "Kurse bleiben unverändert; es fallen nur Kauf- und Verkaufskosten an.",
        "Positives Szenario": f"Jede Aktie steigt um eine Standardabweichung über {tage} Handelstage, "
                              "höchstens bis zum Kursziel.",
    }
    zeilen = []
    for name in beschreibung:
        wert = portfolio.liquiditaet
        for _, pos in portfolio.positionen.iterrows():
            b = nach_ticker[pos["Ticker"]]
            sigma = (b.vola_pct / 100 / math.sqrt(HANDELSTAGE_PRO_JAHR)) * math.sqrt(tage) if _ist_zahl(b.vola_pct) else 0.1
            einstieg = pos["Kurs (verwendet)"]
            if name.startswith("Vorsichtig"):
                preis = max(pos["Stop-Loss"], einstieg * (1 - sigma))
            elif name.startswith("Positiv"):
                preis = min(pos["Kursziel"], einstieg * (1 + sigma))
            else:
                preis = einstieg
            wert += pos["Stück"] * preis * (1 - slip) * (1 - gebuehr)
        ergebnis = wert - portfolio.budget
        nach_steuer = ergebnis - max(ergebnis, 0) * steuer
        zeilen.append({"Szenario": name, "Annahme": beschreibung[name], "Portfolio-Wert (TL)": wert,
                       "Gewinn/Verlust (TL)": ergebnis, "Ergebnis": ergebnis / portfolio.budget,
                       "Gewinn/Verlust nach Steuern (TL)": nach_steuer})
    return pd.DataFrame(zeilen)


# --- Oberfläche: Budget & Aktienauswahl ---------------------------------------------------------------------

ANSICHT_EINZEL = "Einzelanalyse"
ANSICHT_BUDGET = "Budget & Aktienauswahl"


def _budget_seitenleiste() -> dict[str, Any]:
    """Seitenleiste der Budgetplanung."""
    sb = st.sidebar
    quellen = [QUELLE_YAHOO, QUELLE_BORSAPY, QUELLE_DEMO]
    standard = st.session_state.get("quelle", QUELLE_YAHOO)
    quelle = sb.selectbox("Datenquelle", quellen, index=quellen.index(standard) if standard in quellen else 0,
                          key="quelle_budget", help="Für die Aktienauswahl werden mehrere Werte geladen "
                                                    "(CSV ist hier nicht möglich).")
    budget = sb.number_input("Wie viel möchtest du investieren? Betrag in TL", min_value=1.0, max_value=1e12,
                             value=100_000.0, step=1_000.0, format="%.2f", key="budget_betrag")
    sb.caption(f"Budget: **{fmt_tl(budget)}**")
    profil_label = sb.radio("Risikoprofil", list(RISIKO_LABEL.values()), index=1, key="budget_profil")
    profil = next(k for k, v in RISIKO_LABEL.items() if v == profil_label)
    horizont = sb.radio("Anlagehorizont", list(HORIZONTE), index=1, horizontal=True, key="budget_horizont",
                        help="kurzfristig ≈ 1 Monat, mittelfristig ≈ 3 Monate, langfristig ≈ 1 Jahr. "
                             "Beeinflusst Gewichtung, ATR-Stop und Chance-Risiko-Verhältnis.")
    universen = [f"{len(STANDARD_AKTIENLISTE)} große BIST-Werte", f"Alle BIST-Aktien ({len(BIST_ALLE_AKTIEN)})",
                 "Eigene Liste"]
    universum = sb.radio("Analysierte Aktien", universen, key="budget_universum",
                         help="„Alle BIST-Aktien“ prüft den gesamten Markt (dauert beim ersten Mal 1–3 Minuten, "
                              "danach 15 Minuten zwischengespeichert).")
    if universum == universen[0]:
        liste = ", ".join(STANDARD_AKTIENLISTE)
    elif universum == universen[1]:
        liste = ", ".join(BIST_ALLE_AKTIEN)
    else:
        liste = sb.text_area("Eigene Liste (Kürzel, durch Komma getrennt)", ", ".join(STANDARD_AKTIENLISTE),
                             key="budget_liste", height=120)
    p, h = RISIKOPROFILE[profil], HORIZONTE[horizont]
    schluessel = f"{profil}_{horizont}"   # neue Standardwerte bei Profil-/Horizontwechsel
    with sb.expander("Risiko & Positionsgrößen"):
        verlust_pos = st.number_input("Max. Verlust pro Position (% des Budgets)", 0.1, 10.0,
                                      float(p["verlust_position"]), 0.1, key=f"b_vp_{schluessel}")
        verlust_port = st.number_input("Max. Gesamtverlust des Portfolios (%)", 0.5, 50.0,
                                       float(p["verlust_portfolio"]), 0.5, key=f"b_vg_{schluessel}")
        atr_mult = st.number_input("Stop-Loss: ATR-Multiplikator", 0.5, 6.0, float(h["atr_multiplikator"]), 0.1,
                                   key=f"b_atr_{schluessel}")
        crv = st.number_input("Chance-Risiko-Verhältnis (Kursziel)", 0.5, 6.0, float(h["crv"]), 0.1,
                              key=f"b_crv_{schluessel}")
        reserve_pct = st.number_input("Liquiditätsreserve (% des Budgets)", 0.0, 95.0, float(p["reserve"]), 1.0,
                                      key=f"b_res_{schluessel}")
        reserve_tl = st.number_input("Mindest-Liquiditätsreserve (TL)", 0.0, 1e12, 0.0, 1_000.0, format="%.2f",
                                     key="b_res_tl")
        min_anteil, max_anteil = st.slider("Anteil je Aktie: Minimum / Maximum (%)", 1, 50,
                                           (int(p["min_anteil"]), int(p["max_anteil"])), key=f"b_ant_{schluessel}",
                                           help="Höchstens 50 % – das Budget fließt nie vollständig in eine Aktie.")
        max_pos = st.slider("Max. Anzahl gleichzeitiger Positionen", 2, 12, int(p["max_positionen"]),
                            key=f"b_pos_{schluessel}")
        max_vola = st.slider("Max. Volatilität p. a. (%) – darüber Ausschluss", 20, 150, int(p["max_vola"]),
                             key=f"b_vola_{schluessel}")
        min_liq = st.number_input("Mindest-Liquidität (Ø Umsatz, Mio. TL/Tag)", 0.0, 10_000.0, 20.0, 5.0,
                                  key="b_liq", help="Darunter Punktabzug, unter einem Fünftel Ausschluss.")
    with sb.expander("Gebühren, Slippage & Steuern"):
        gebuehren = st.number_input("Gebühren je Order (%)", 0.0, 3.0, 0.10, 0.01, format="%.2f", key="b_geb")
        slippage = st.number_input("Slippage je Ausführung (%)", 0.0, 3.0, 0.05, 0.01, format="%.2f", key="b_slip")
        steuer = st.number_input("Steuer auf Kursgewinne (%)", 0.0, 60.0, 0.0, 0.5, key="b_steuer",
                                 help="Nur für die Szenarien. Bitte die aktuell gültige Regelung für deine Situation "
                                      "prüfen.")
    bp = BudgetParameter.aus_profil(
        float(budget), profil, horizont, verlust_position_pct=float(verlust_pos),
        verlust_portfolio_pct=float(verlust_port), atr_multiplikator=float(atr_mult), crv=float(crv),
        reserve_pct=float(reserve_pct), reserve_min_tl=float(reserve_tl), min_anteil_pct=float(min_anteil),
        max_anteil_pct=float(max_anteil), max_positionen=int(max_pos), max_vola_pct=float(max_vola),
        min_liquiditaet_tl=float(min_liq) * 1e6, gebuehren_pct=float(gebuehren), slippage_pct=float(slippage),
        steuer_pct=float(steuer))
    ticker = [t.strip() for t in re.split(r"[,;\s]+", liste) if t.strip()]
    return {"quelle": quelle, "bp": bp, "ticker": list(dict.fromkeys(ticker))}


def screening_durchfuehren(ticker_liste: Sequence[str], quelle: str, bp: BudgetParameter,
                           jetzt: pd.Timestamp | None = None, fortschritt: Any = None) -> list[AktienBewertung]:
    """Lädt Kurse (bei Yahoo gebündelt) und bewertet alle Aktien der Liste."""
    jetzt = jetzt if jetzt is not None else _jetzt_istanbul()
    live = quelle in LIVE_QUELLEN
    info = INTERVALLE["Täglich"]
    start = (jetzt - pd.Timedelta(days=ZEITRAEUME["1 Jahr"] + info.vorlauf_tage)).normalize()
    ende = jetzt.normalize() + pd.Timedelta(days=1)
    vorrat: dict[str, pd.DataFrame] = {}
    if quelle == QUELLE_YAHOO:
        basis_liste = []
        for eingabe in ticker_liste:
            with suppress(DatenFehler):
                basis_liste.append(ticker_normalisieren(eingabe).basis)
        vorrat = yahoo_mehrere_laden(basis_liste, start, ende)

    def laden(basis: str) -> DatenPaket:
        if quelle == QUELLE_YAHOO:
            if basis not in vorrat:
                raise DatenFehler(f"Keine Kursdaten bei Yahoo Finance für {basis}{STANDARD_SUFFIX}.",
                                  art="keine_daten")
            return DatenPaket(daten=vorrat[basis], ticker=ticker_normalisieren(basis),
                              symbol=f"{basis}{STANDARD_SUFFIX}", quelle=quelle, intervall="Täglich",
                              zeitraum="1 Jahr", anzeige_start=start, waehrung="TRY",
                              name=BIST_FAVORITEN.get(basis) or BIST_ALLE_AKTIEN.get(basis, basis),
                              ist_index=basis in BIST_INDIZES)
        return daten_laden(basis, "1 Jahr", "Täglich", quelle, jetzt=jetzt)

    return aktien_screenen(ticker_liste, laden, bp, StrategieParameter(), live=live, jetzt=jetzt,
                           fortschritt=fortschritt)


@_cache_data(ttl=900)
def _screening_gecacht(ticker: tuple[str, ...], quelle: str, horizont: str, max_vola: float, min_liq: float,
                       atr_mult: float, crv: float, stunde: str) -> list[AktienBewertung]:
    """Zwischengespeicherte Bewertung (unabhängig von Budget und Aufteilungsparametern)."""
    bp = BudgetParameter.aus_profil(1.0, "mittel", horizont, max_vola_pct=max_vola, min_liquiditaet_tl=min_liq,
                                    atr_multiplikator=atr_mult, crv=crv)
    return screening_durchfuehren(list(ticker), quelle, bp)


def _bewertungstabelle(bewertungen: Sequence[AktienBewertung]) -> pd.DataFrame:
    zeilen = []
    for rang, b in enumerate([x for x in bewertungen if x.ok], start=1):
        zeilen.append({
            "Rang": rang, "Aktie": f"{b.name} ({b.ticker})", "Signal": b.signal,
            "Score": f"{fmt_zahl(b.punkte, 1, True)} Pkt. ({fmt_pct(b.score_pct, 0, True)})",
            "Kurs": b.kurs, "Veränderung": b.veraenderung_pct, "Risiko": b.risiko_stufe,
            "Begründung": "; ".join(b.gruende[:2] + [f"Risiko: {r}" for r in b.risiken[:1]]) or "–",
        })
    return pd.DataFrame(zeilen, columns=["Rang", "Aktie", "Signal", "Score", "Kurs", "Veränderung", "Risiko",
                                         "Begründung"])


def budget_ansicht_starten() -> None:
    """Budgetplanung, Aktienauswahl, Marktüberblick, Aufteilung und Szenarien."""
    e = _budget_seitenleiste()
    bp: BudgetParameter = e["bp"]
    st.title("Budgetplanung & aktuelle Kaufbewertung")
    st.caption(BUDGET_TITEL)
    st.error(BUDGET_WARNHINWEIS, icon=":material/gavel:")
    fehler = bp.pruefen()
    if fehler:
        for text in fehler:
            st.error(text)
        st.stop()
    if not e["ticker"]:
        st.warning("Bitte mindestens ein Börsenkürzel in der Seitenleiste eingeben.")
        st.stop()

    quelle = e["quelle"]
    jetzt = _jetzt_istanbul()
    with st.spinner(f"Lade und bewerte {len(e['ticker'])} Aktien … (bei allen Aktien 1–3 Minuten)"):
        bewertungen = _screening_gecacht(tuple(e["ticker"]), quelle, bp.horizont, bp.max_vola_pct,
                                         bp.min_liquiditaet_tl, bp.atr_multiplikator, bp.crv,
                                         jetzt.strftime("%Y-%m-%d %H"))
    gueltig = [b for b in bewertungen if b.ok]
    ausgeschlossen = [b for b in bewertungen if not b.ok]
    portfolio = portfolio_aufteilen(bewertungen, bp)
    szenarien = szenarien_berechnen(portfolio, bewertungen, bp)
    erstellt = datetime.now()
    verzoegert = {QUELLE_YAHOO: "verzögert (in der Regel ≥ 15 Minuten), keine Echtzeitdaten",
                  QUELLE_BORSAPY: "verzögert (ca. 15 Minuten), keine Echtzeitdaten",
                  QUELLE_DEMO: "DEMO – synthetische Kurse, keine Marktdaten"}[quelle]
    if quelle == QUELLE_DEMO:
        st.error("DEMO-MODUS: synthetische Kurse – für echte Werte die Datenquelle Yahoo Finance wählen.",
                 icon=":material/science:")
    if not gueltig:
        st.error("Für keine Aktie liegen ausreichende, aktuelle Kursdaten vor. Details unter „Transparenz“.")

    # ---------------------------------------------------------------- 10) Zusammenfassung
    st.subheader("Zusammenfassung")
    staerkste = next((b for b in gueltig if b.signal in (SCREEN_STARK_KAUF, SCREEN_KAUF)), None)
    a, b_, c, d = st.columns(4)
    a.metric("Budget", fmt_tl(bp.budget), border=True)
    b_.metric("Risikoprofil / Horizont", f"{bp.risikoprofil} · {bp.horizont}", border=True)
    c.metric("Vorgeschlagene Positionen", str(len(portfolio.positionen)), border=True)
    d.metric("Liquidität (inkl. Reserve)", fmt_tl(portfolio.liquiditaet, 0), border=True,
             help=f"Davon Reserve {fmt_tl(portfolio.reserve, 0)}; der Rest ist nicht investierbar gewesen.")
    a, b_, c, d = st.columns(4)
    a.metric("Investiert (Beispiel)", fmt_tl(portfolio.investiert, 0), border=True)
    b_.metric("Max. berechnetes Risiko", fmt_tl(portfolio.max_verlust, 0), border=True,
              help="Summe der Verluste, falls alle Positionen ihren Stop-Loss erreichen (inkl. Kosten).")
    c.metric("Risiko in % des Budgets", fmt_pct(portfolio.max_verlust / bp.budget, 2, False), border=True)
    if len(portfolio.positionen):
        durchschnitt = float(np.average(portfolio.positionen["Score"], weights=portfolio.positionen["Betrag (TL)"]))
        d.metric("Ø Score des Beispielportfolios", f"{fmt_zahl(durchschnitt, 1)} %", border=True)
    else:
        d.metric("Ø Score des Beispielportfolios", "–", border=True)
    if staerkste is not None:
        st.success(f"**{staerkste.name} ({staerkste.ticker}):** {TOP_FORMULIERUNG} Score "
                   f"{fmt_zahl(staerkste.punkte, 1, True)} Punkte ({fmt_pct(staerkste.score_pct, 0)}), Signal "
                   f"„{staerkste.signal}“.", icon=":material/insights:")
    else:
        st.info("Nach den ausgewählten technischen Kriterien zeigt derzeit keine Aktie ein positives Signal.")
    for hinweis in portfolio.hinweise:
        st.info(hinweis, icon=":material/info:")

    # ---------------------------------------------------------------- 6) Marktüberblick
    st.subheader("Marktüberblick für den ausgewählten Handelstag")
    status = markt_status(jetzt)
    handelstag = erwarteter_handelstag(jetzt)
    aktuelle = sum(b.aktuell for b in gueltig)
    letzte = max((b.datum for b in gueltig if b.datum is not None), default=None)
    a, b_, c = st.columns(3)
    a.metric("Borsa İstanbul (Uhrzeit Istanbul)", status["text"], border=True, help=status["hinweis"])
    b_.metric("Letzte Kursdaten", fmt_datum(letzte) if letzte is not None else "–", border=True,
              help=f"Erwarteter Handelstag: {fmt_datum(handelstag)}. Datenqualität: {verzoegert}.")
    c.metric("Daten aktuell", f"{aktuelle} von {len(gueltig)} Aktien", border=True)
    st.caption(f"Handelstag: {fmt_datum(handelstag)} · Analyse erstellt: {erstellt.strftime('%d.%m.%Y %H:%M')} · "
               f"Daten: {verzoegert}.")
    tabelle = _bewertungstabelle(bewertungen)
    st.dataframe(tabelle.style.map(_farbe_fuer_text, subset=["Signal"])
                 .map(_farbe_fuer_zahl, subset=["Veränderung"])
                 .format({"Kurs": lambda v: fmt_tl(v), "Veränderung": lambda v: fmt_pct(v)}),
                 hide_index=True, height=_tabellenhoehe(min(len(tabelle), 20)))
    if len(tabelle) > 20:
        st.caption(f"{len(tabelle)} bewertete Aktien – in der Tabelle scrollen. Sortierung nach Score.")
    neu = [b for b in gueltig if b.neu]
    fortgesetzt = [b for b in gueltig if not b.neu and b.signal != SCREEN_BEOBACHTEN]

    def gekuerzt(zeilen: list[str], anzahl: int = 25) -> str:
        if not zeilen:
            return "– keine –"
        rest = f"\n- … und {len(zeilen) - anzahl} weitere" if len(zeilen) > anzahl else ""
        return "\n".join(zeilen[:anzahl]) + rest

    x, y, z = st.columns(3)
    with x:
        st.markdown("**Neues Signal heute**")
        st.markdown(gekuerzt([f"- {b.ticker}: {b.signal} (vorher: {b.signal_vortag})" for b in neu]))
    with y:
        st.markdown("**Bestehendes Signal fortgesetzt**")
        st.markdown(gekuerzt([f"- {b.ticker}: {b.signal}" for b in fortgesetzt]))
    with z:
        st.markdown("**Nicht berücksichtigen (Volatilität, Daten)**")
        st.markdown(gekuerzt([f"- {b.ticker}: {b.ausschlussgrund}" for b in ausgeschlossen], 15)
                    + ("\n\nVollständige Liste unter „Transparenz“." if len(ausgeschlossen) > 15 else ""))

    # ---------------------------------------------------------------- 3) Aktuelle Signale je Aktie
    st.subheader("Aktuelle Signale je Aktie")
    st.caption(f"Signalstufen nach Score in % der erreichbaren Punkte: ≥ 50 % {SCREEN_STARK_KAUF}, ≥ 25 % "
               f"{SCREEN_KAUF}, ≤ −25 % {SCREEN_VERKAUF}, ≤ −50 % {SCREEN_STARK_VERKAUF}, sonst {SCREEN_BEOBACHTEN}.")
    auswahl_details = gueltig[:25]
    if len(gueltig) > 25:
        weitere = st.multiselect("Weitere Aktien im Detail anzeigen", [b.ticker for b in gueltig[25:]],
                                 key="budget_details",
                                 help="Die 25 bestbewerteten Aktien werden immer angezeigt.")
        auswahl_details += [b for b in gueltig[25:] if b.ticker in weitere]
        st.caption(f"Angezeigt: die 25 bestbewerteten von {len(gueltig)} Aktien (weitere oben auswählbar).")
    for b in auswahl_details:
        titel = (f"{b.ticker} · {b.name} — {b.signal} · Score {fmt_zahl(b.punkte, 1, True)} Pkt. "
                 f"({fmt_pct(b.score_pct, 0)}) · {fmt_tl(b.kurs)} ({fmt_pct(b.veraenderung_pct)})")
        with st.expander(titel):
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Kurs", fmt_tl(b.kurs), fmt_pct(b.veraenderung_pct), border=True)
            k2.metric("ATR-Stop-Loss", fmt_tl(b.stop), fmt_pct(b.stop / b.kurs - 1), border=True)
            k3.metric("Mögliches Kursziel", fmt_tl(b.ziel), fmt_pct(b.ziel / b.kurs - 1), border=True)
            k4.metric("Chance-Risiko-Verhältnis", fmt_zahl(b.crv, 1), border=True,
                      help=f"Bis zum nächsten Widerstand: {fmt_zahl(b.crv_bis_widerstand, 1)}")
            zone_u = (f"{fmt_zahl(b.unterstuetzung.unten)}–{fmt_zahl(b.unterstuetzung.oben)}"
                      if b.unterstuetzung else "keine erkannt")
            zone_w = (f"{fmt_zahl(b.widerstand.unten)}–{fmt_zahl(b.widerstand.oben)}"
                      if b.widerstand else "keine erkannt")
            st.markdown(
                f"- **Signalstärke:** {fmt_zahl(b.punkte, 1, True)} von max. {fmt_zahl(b.max_punkte, 1)} Punkten "
                f"({fmt_pct(b.score_pct, 0)}) · Regel-Ereignissignal (MACD): {b.ereignissignal}\n"
                f"- **Gründe:** {'; '.join(b.gruende) or '–'}\n"
                f"- **Risiken:** {'; '.join(b.risiken) or '–'}\n"
                f"- **Unterstützungszone:** {zone_u} · **Widerstandszone:** {zone_w}\n"
                f"- **Volatilität:** {fmt_zahl(b.vola_pct, 0)} % p. a. · **Ø Umsatz:** "
                f"{fmt_volumen(b.liquiditaet_tl)} TL/Tag · **Risikostufe:** {b.risiko_stufe}\n"
                f"- **Letzte Kursdaten:** {fmt_datum(b.datum)}{' (vorläufig, Handel läuft)' if b.vorlaeufig else ''}"
                f" · {verzoegert}")

    # ---------------------------------------------------------------- 4/5) Budgetaufteilung
    st.subheader("Beispielhafte Budgetaufteilung (risikobasiert)")
    st.caption(f"Stückzahl = min(max. Verlust je Position ÷ Positionsrisiko, Zielbetrag ÷ Kurs inkl. Gebühren); "
               f"Positionsrisiko = Einstiegskurs − Stop-Loss ({fmt_zahl(bp.atr_multiplikator, 1)} × ATR) zzgl. "
               f"Kosten. Nur ganze Aktien. Anteil je Aktie {fmt_zahl(bp.min_anteil_pct, 0)}–"
               f"{fmt_zahl(bp.max_anteil_pct, 0)} %, max. {bp.max_positionen} Positionen, Reserve "
               f"{fmt_pct(portfolio.reserve / bp.budget, 0, False)}.")
    if len(portfolio.positionen):
        geld = {s: (lambda v: fmt_tl(v)) for s in ("Betrag (TL)", "Kurs (verwendet)", "Restbetrag (TL)",
                                                     "Stop-Loss", "Kursziel", "Max. Verlust bis Stop (TL)",
                                                     "Gebühren (TL)")}
        st.dataframe(portfolio.positionen.style.map(_farbe_fuer_text, subset=["Signal"])
                     .format({**geld, "Anteil": lambda v: fmt_pct(v, 1, False), "Score": lambda v: f"{fmt_zahl(v, 1)} %"}),
                     hide_index=True)
        st.markdown("**Gründe und Risiken je Position**\n" + "\n".join(
            f"- **{pos['Ticker']}** ({pos['Stück']} Stück · {fmt_tl(pos['Betrag (TL)'], 0)} · Stop "
            f"{fmt_tl(pos['Stop-Loss'])} · Ziel {fmt_tl(pos['Kursziel'])} · max. Verlust "
            f"{fmt_tl(pos['Max. Verlust bis Stop (TL)'], 0)})  \n  Gründe: {pos['Gründe']}  \n  Risiken: {pos['Risiken']}"
            for _, pos in portfolio.positionen.iterrows()))
        st.markdown(f"**Investiert:** {fmt_tl(portfolio.investiert)} · **Gebühren (Kauf):** "
                    f"{fmt_tl(portfolio.gebuehren)} · **Liquidität gesamt:** {fmt_tl(portfolio.liquiditaet)} "
                    f"(Reserve {fmt_tl(portfolio.reserve)} + nicht verwendeter Restbetrag)")
    else:
        st.info("Keine Positionen im Beispielportfolio.")

    # ---------------------------------------------------------------- 7) Szenarien
    st.subheader("Drei Szenarien")
    st.warning("Die Szenarien sind keine Prognose und keine Garantie. Sie zeigen nur, wie sich das Beispielportfolio "
               "unter vereinfachten Annahmen entwickeln könnte.", icon=":material/warning:")
    st.dataframe(szenarien.drop(columns=["Annahme"]).style
                 .map(_farbe_fuer_zahl, subset=["Gewinn/Verlust (TL)", "Ergebnis"])
                 .format({"Portfolio-Wert (TL)": fmt_tl, "Gewinn/Verlust (TL)": fmt_tl,
                          "Gewinn/Verlust nach Steuern (TL)": fmt_tl, "Ergebnis": lambda v: fmt_pct(v)}),
                 hide_index=True)
    st.markdown("**Zugrunde liegende Annahmen**\n" + "\n".join(
        f"- **{z['Szenario']}:** {z['Annahme']}" for _, z in szenarien.iterrows())
        + "\n- Volatilität je Aktie aus den letzten 20 Handelstagen; Verkaufskosten und Slippage sind abgezogen; "
          "die Liquidität bleibt unverändert (ohne Zinsen).")

    # ---------------------------------------------------------------- 8) Transparenz
    st.subheader("Transparenz")
    gewichte = HORIZONTE[bp.horizont]["gewichte"]
    st.markdown(
        f"- **Analysierte Aktien ({len(bewertungen)}):** {', '.join(b.ticker for b in bewertungen)}\n"
        f"- **Datenquelle:** {quelle} · **Datenqualität:** {verzoegert}\n"
        f"- **Analyse erstellt:** {erstellt.strftime('%d.%m.%Y %H:%M:%S')} (Istanbul: "
        f"{jetzt.strftime('%d.%m.%Y %H:%M')})\n"
        f"- **Gebühren:** {fmt_zahl(bp.gebuehren_pct, 2)} % je Order berücksichtigt · **Slippage:** "
        f"{fmt_zahl(bp.slippage_pct, 2)} % berücksichtigt · **Steuern:** "
        + (f"{fmt_zahl(bp.steuer_pct, 1)} % auf Gewinne in den Szenarien" if bp.steuer_pct else
           "nicht berücksichtigt (0 %) – bitte selbst prüfen")
        + "\n- **Einflussfaktoren (Gewichte für " + bp.horizont + "):** "
        + ", ".join(f"{KOMPONENTEN_NAMEN[k]} × {fmt_zahl(v, 1)}" for k, v in gewichte.items())
        + f", {KOMPONENTEN_NAMEN['zonen']} ±1, {KOMPONENTEN_NAMEN['stop']} −1, {KOMPONENTEN_NAMEN['liquiditaet']} −1")
    if ausgeschlossen:
        st.markdown("**Ausgeschlossene Aktien**")
        st.dataframe(pd.DataFrame([{"Aktie": b.ticker, "Grund": b.ausschlussgrund} for b in ausgeschlossen]),
                     hide_index=True)
    if gueltig:
        with st.expander("Beitrag jeder Komponente zum Score"):
            beitraege = pd.DataFrame([{"Aktie": b.ticker, **{KOMPONENTEN_NAMEN[k]: v for k, v in b.komponenten.items()},
                                       "Summe": b.punkte} for b in gueltig])
            st.dataframe(beitraege.style.format(lambda v: fmt_zahl(v, 1, True), subset=beitraege.columns[1:]),
                         hide_index=True)
    a, b_ = st.columns(2)
    a.download_button("Bewertungen (CSV)", als_csv(tabelle, True, index=False), "bist_bewertungen.csv", "text/csv",
                      on_click="ignore", icon=":material/download:")
    b_.download_button("Beispielportfolio (CSV)", als_csv(portfolio.positionen, True, index=False),
                       "bist_beispielportfolio.csv", "text/csv", on_click="ignore", icon=":material/download:")
    st.divider()
    st.caption(f"{BUDGET_TITEL} {BUDGET_WARNHINWEIS}")


# =============================================================================
# 15) Programmstart
# =============================================================================

if __name__ == "__main__":
    if _laeuft_in_streamlit():
        streamlit_app_starten()
    else:  # Start per "python bist_analyse_app.py" → Streamlit-Server starten
        try:
            from streamlit.web import cli as streamlit_cli
        except ImportError:
            print("Streamlit ist nicht installiert. Bitte zuerst ausführen: pip install -r requirements.txt")
            sys.exit(1)
        sys.argv = ["streamlit", "run", str(Path(__file__).resolve()), *sys.argv[1:]]
        sys.exit(streamlit_cli.main())

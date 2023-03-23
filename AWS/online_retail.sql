use synan_dsmlbc_db;

# query 1
# musterinin toplam alisveris sayisi
# bir seferde birden cok urun alsa bile bu 1 sayilacaktir.
# fatura numarasina gore gruplama yaptigimiz icin
# query 1

SELECT CustomerID,
       COUNT(*) as totalSales
FROM online_retail_2009_2010
# Musteri numarasi olmayan kayitlari cikardik
WHERE CustomerID is not null
#  buradada geri iade leri cikardik
  and Quantity * Price >= 0
GROUP BY Invoice, CustomerID
ORDER BY totalSales DESC;


# query 2
# musterilerin toplam yaptigi harcama
# query 2
SELECT CustomerID, ROUND(SUM(Price * Quantity)) as total
FROM online_retail_2009_2010
WHERE CustomerID is not null AND Quantity * Price >= 0
GROUP BY CustomerID
ORDER BY total DESC;

# query 3
# Sitede Yapilan Aylik Satis
# query 3
SELECT YEAR(InvoiceDate)            as Year,
       MONTH(InvoiceDate)           as Month,
       ROUND(SUM(Price * Quantity)) AS TotalSales
FROM online_retail_2009_2010
GROUP BY Year, Month
ORDER BY Year, Month;

# query 4
#  musteri numarasina gore aggregation
#  haftasonu ve haftaici alisveris tutari ve toplam tutar
#  DAYOFWEEK() fonksiyonu tek hane digit olarak haftanin hangi gunu oldugunu verir. dikkat pazardan basliyor
#  DAYOFWEEK 1=Sunday, 2=Monday, 3=Tuesday, 4=Wednesday, 5=Thursday, 6=Friday, 7=Saturday.
#  UNION iki farkli queryi birlestirmeye yariyor . sadece union yazarsak distinct davranir Bu yuzden UNION ALL yazdik
#  UNION Kullanirken birlestirdigimiz query nin select teki column sayisi ve sirasi ayni olmali bu yuzden bos kalan yerlere 0 olarak tanimladik (0 AS weekdays )
# query 4
SELECT CustomerID,
       ROUND(SUM(A.weekend))              AS weekend_amount,
       ROUND(SUM(A.weekdays))             AS weekday_amount,
       ROUND(SUM(A.weekdays + A.weekend)) AS total_amount
FROM (
         SELECT CustomerID,
                SUM(Price * Quantity) AS weekend,
                0                     AS weekdays
         FROM online_retail_2009_2010
         WHERE DAYOFWEEK(InvoiceDate) IN (1, 7)
           AND CustomerID is not null
           AND Quantity * Price >= 0
         GROUP BY CustomerID
         UNION ALL

         SELECT CustomerID,
                0                     AS weekend,
                SUM(Price * Quantity) AS weekdays
         FROM online_retail_2009_2010
         WHERE DAYOFWEEK(InvoiceDate) NOT IN (1, 7)
           AND CustomerID is not null
           AND Quantity * Price >= 0
         GROUP BY CustomerID) as A
GROUP BY CustomerID
ORDER BY total_amount DESC;


# query 5
# query 4'e biraz daha senlik getirip haftasonu ve haftaici yapilan alisverisin toplan alisverise oranini ekledik
# ve anladik ki haftasonu insanlar eglencede, gezmede, tozmada. online alisveris yapan yok
# query 5
SELECT CustomerID,
       SUM(A.weekend)                                                  AS weekend_total,
       ROUND(SUM(A.weekdays))                                          AS weekday_total,
       ROUND(SUM(A.weekdays + weekend))                                AS total,
       ROUND((SUM(A.weekend) / SUM(A.weekdays + A.weekend) * 100), 2)  AS weekend_percentage,
       ROUND((SUM(A.weekdays) / SUM(A.weekdays + A.weekend) * 100), 2) AS weekday_percentage

FROM (
         SELECT CustomerID,
                ROUND(SUM(Price * Quantity)) AS weekend,
                0                            AS weekdays
         FROM online_retail_2009_2010

         WHERE DAYOFWEEK(InvoiceDate) IN (1, 7)
           AND CustomerID is not null
           AND Quantity * Price >= 0
         GROUP BY CustomerID
         UNION ALL
         SELECT CustomerID,
                0                     AS weekend,
                SUM(Price * Quantity) AS weekdays
         FROM online_retail_2009_2010
         WHERE DAYOFWEEK(InvoiceDate) NOT IN (1, 7)
           AND CustomerID is not null
           AND Quantity * Price >= 0
         GROUP BY CustomerID) as A

GROUP BY CustomerID
ORDER BY total DESC
LIMIT 10000;



# query 6
# Burada haftasonu ve hafta ici kismini kaldirip haftanin gunlerine gore olay yerini inceledik.
# monday_p -> pazartesi gunu yapilan alisverisin haftanin diger gunlarinin toplam alisverise yuzdelik orani ( percantage )
# query 6

SELECT B.CustomerID                              as CustomerID,
       B.total_amount                            as total_amount,
       B.monday,
       ROUND(B.monday / B.total_amount * 100)    as moday_p,
       B.tuesday,
       ROUND(B.tuesday / B.total_amount * 100)   as tuesday_p,
       B.wednesday,
       ROUND(B.wednesday / B.total_amount * 100) as wednesday_p,
       B.thursday,
       ROUND(B.thursday / B.total_amount * 100)  as thursday_p,
       B.friday,
       ROUND(B.friday / B.total_amount * 100)    as friday_p,
       B.saturday,
       ROUND(B.saturday / B.total_amount * 100)  as saturday_p,
       B.sunday,
       ROUND(B.sunday / B.total_amount * 100)    as sunday_p
FROM (
         SELECT CustomerID,
                ROUND(SUM(monday))    as monday,
                ROUND(SUM(tuesday))   as tuesday,
                ROUND(SUM(wednesday)) as wednesday,
                ROUND(SUM(thursday))  as thursday,
                ROUND(SUM(friday))    as friday,
                ROUND(SUM(saturday))  as saturday,
                ROUND(SUM(sunday))    as sunday,
                ROUND(SUM(monday
                    + tuesday
                    + wednesday
                    + thursday
                    + friday
                    + saturday
                    + sunday))        AS total_amount

         FROM (
                  SELECT CustomerID,
                         ROUND(SUM(Price * Quantity)) AS monday,
                         0                            as tuesday,
                         0                            as wednesday,
                         0                            as thursday,
                         0                            as friday,
                         0                            as saturday,
                         0                            as sunday
                  FROM online_retail_2009_2010
                  WHERE DAYOFWEEK(InvoiceDate) IN (2)
                    AND CustomerID is not null
                    AND Quantity * Price >= 0
                  GROUP BY CustomerID
                  UNION ALL

                  SELECT CustomerID,
                         0                            as monday,
                         ROUND(SUM(Price * Quantity)) AS tuesday,
                         0                            as wednesday,
                         0                            as thursday,
                         0                            as friday,
                         0                            as saturday,
                         0                            as sunday
                  FROM online_retail_2009_2010
                  WHERE DAYOFWEEK(InvoiceDate) IN (3)
                    AND CustomerID is not null
                    AND Quantity * Price >= 0
                  GROUP BY CustomerID
                  UNION ALL

                  SELECT CustomerID,
                         0                            as monday,
                         0                            as tuesday,
                         ROUND(SUM(Price * Quantity)) AS wednesday,
                         0                            as thursday,
                         0                            as friday,
                         0                            as saturday,
                         0                            as sunday
                  FROM online_retail_2009_2010
                  WHERE DAYOFWEEK(InvoiceDate) IN (4)
                    AND CustomerID is not null
                    AND Quantity * Price >= 0
                  GROUP BY CustomerID
                  UNION ALL

                  SELECT CustomerID,
                         0                            as monday,
                         0                            as tuesday,
                         0                            as wednesday,
                         ROUND(SUM(Price * Quantity)) AS thursday,
                         0                            as friday,
                         0                            as saturday,
                         0                            as sunday
                  FROM online_retail_2009_2010
                  WHERE DAYOFWEEK(InvoiceDate) IN (5)
                    AND CustomerID is not null
                    AND Quantity * Price >= 0
                  GROUP BY CustomerID
                  UNION ALL

                  SELECT CustomerID,
                         0                            as monday,
                         0                            as tuesday,
                         0                            as wednesday,
                         0                            as thursday,
                         ROUND(SUM(Price * Quantity)) AS friday,


                         0                            as saturday,
                         0                            as sunday
                  FROM online_retail_2009_2010
                  WHERE DAYOFWEEK(InvoiceDate) IN (6)
                    AND CustomerID is not null
                    AND Quantity * Price >= 0
                  GROUP BY CustomerID
                  UNION ALL

                  SELECT CustomerID,
                         0                            as monday,
                         0                            as tuesday,
                         0                            as wednesday,
                         0                            as thursday,
                         0                            as friday,
                         ROUND(SUM(Price * Quantity)) AS saturday,


                         0                            as sunday
                  FROM online_retail_2009_2010
                  WHERE DAYOFWEEK(InvoiceDate) IN (7)
                    AND CustomerID is not null
                    AND Quantity * Price >= 0
                  GROUP BY CustomerID
                  UNION ALL

                  SELECT CustomerID,
                         0                            as monday,
                         0                            as tuesday,
                         0                            as wednesday,
                         0                            as thursday,
                         0                            as friday,
                         0                            as saturday,
                         ROUND(SUM(Price * Quantity)) AS sunday


                  FROM online_retail_2009_2010
                  WHERE DAYOFWEEK(InvoiceDate) IN (1)
                    and Price * Quantity > -1
                  GROUP BY CustomerID
              ) as A
         GROUP BY CustomerID
     ) as B

GROUP BY CustomerID
ORDER BY B.total_amount DESC
LIMIT 10000;

# query 7
# musteri numarasina gore gruplamayi (aggregation) biraktik onun yerine yillara gore grupladik
# query 7

SELECT year,
       ROUND(B.total),
       monday,
       ROUND(monday/B.total*100)  as monday_p,
       tuesday,
       ROUND(tuesday/B.total*100)  as tuesday_p,
       wednesday,
       ROUND(wednesday/B.total*100)  as wednesday_p,
       thursday,
       ROUND(thursday/B.total*100)  as thursday_p,
       friday,
       ROUND(friday/B.total*100)  as friday_p,
       saturday,
       ROUND(saturday/B.total*100)  as saturday_p,
       sunday,
       ROUND(sunday/B.total*100)  as sunday_p
FROM (
         SELECT A.year,
                ROUND(SUM(A.monday))    as monday,
                ROUND(SUM(A.tuesday))   as tuesday,
                ROUND(SUM(A.wednesday)) as wednesday,
                ROUND(SUM(A.thursday))  as thursday,
                ROUND(SUM(A.friday))    as friday,
                ROUND(SUM(A.saturday))  as saturday,
                ROUND(SUM(A.sunday))    as sunday,
                SUM(  A.monday
                    + A.tuesday
                    + A.wednesday
                    + A.thursday
                    + A.friday
                    + A.saturday
                    + A.sunday)         as total

         FROM (
                  SELECT YEAR(InvoiceDate)     as year,
                         SUM(Price * Quantity) AS monday,
                         0                     as tuesday,
                         0                     as wednesday,
                         0                     as thursday,
                         0                     as friday,
                         0                     as saturday,
                         0                     as sunday
                  FROM online_retail_2009_2010
                  WHERE DAYOFWEEK(InvoiceDate) IN (2)
                    AND CustomerID is not null
                    AND Quantity * Price >= 0
                  GROUP BY year
                  UNION ALL

                  SELECT YEAR(InvoiceDate)     as year,
                         0                     as monday,
                         SUM(Price * Quantity) AS tuesday,
                         0                     as wednesday,
                         0                     as thursday,
                         0                     as friday,
                         0                     as saturday,
                         0                     as sunday
                  FROM online_retail_2009_2010
                  WHERE DAYOFWEEK(InvoiceDate) IN (3)
                    AND CustomerID is not null
                    AND Quantity * Price >= 0
                  GROUP BY year
                  UNION ALL

                  SELECT YEAR(InvoiceDate)     as year,
                         0                     as monday,
                         0                     as tuesday,
                         SUM(Price * Quantity) AS wednesday,
                         0                     as thursday,
                         0                     as friday,
                         0                     as saturday,
                         0                     as sunday
                  FROM online_retail_2009_2010
                  WHERE DAYOFWEEK(InvoiceDate) IN (4)
                    AND CustomerID is not null
                    AND Quantity * Price >= 0
                  GROUP BY year
                  UNION ALL

                  SELECT YEAR(InvoiceDate)     as year,
                         0                     as monday,
                         0                     as tuesday,
                         0                     as wednesday,
                         SUM(Price * Quantity) AS thursday,
                         0                     as friday,
                         0                     as saturday,
                         0                     as sunday
                  FROM online_retail_2009_2010
                  WHERE DAYOFWEEK(InvoiceDate) IN (5)
                    AND CustomerID is not null
                    AND Quantity * Price >= 0
                  GROUP BY year
                  UNION ALL

                  SELECT YEAR(InvoiceDate)     as year,
                         0                     as monday,
                         0                     as tuesday,
                         0                     as wednesday,
                         0                     as thursday,
                         SUM(Price * Quantity) AS friday,
                         0                     as saturday,
                         0                     as sunday
                  FROM online_retail_2009_2010
                  WHERE DAYOFWEEK(InvoiceDate) IN (6)
                    AND CustomerID is not null
                    AND Quantity * Price >= 0
                  GROUP BY year
                  UNION ALL

                  SELECT YEAR(InvoiceDate)     as year,
                         0                     as monday,
                         0                     as tuesday,
                         0                     as wednesday,
                         0                     as thursday,
                         0                     as friday,
                         SUM(Price * Quantity) AS saturday,


                         0                     as sunday
                  FROM online_retail_2009_2010
                  WHERE DAYOFWEEK(InvoiceDate) IN (7)
                    AND CustomerID is not null
                    AND Quantity * Price >= 0
                  GROUP BY year
                  UNION ALL

                  SELECT YEAR(InvoiceDate)     as year,
                         0                     as monday,
                         0                     as tuesday,
                         0                     as wednesday,
                         0                     as thursday,
                         0                     as friday,
                         0                     as saturday,
                         SUM(Price * Quantity) AS sunday


                  FROM online_retail_2009_2010
                  WHERE DAYOFWEEK(InvoiceDate) IN (1)
                    AND CustomerID is not null
                    AND Quantity * Price >= 0
                  GROUP BY year
              ) as A
     GROUP BY year) as B


GROUP BY year
LIMIT 10000;


# Yazar: Sinan Artun
#      _               _ _            ___
#     | |             | | |          /   |
#   __| |___ _ __ ___ | | |__   ___ / /| |
#  / _` / __| '_ ` _ \| | '_ \ / __/ /_| |
# | (_| \__ \ | | | | | | |_) | (__\___  |
#  \__,_|___/_| |_| |_|_|_.__/ \___|   |_/
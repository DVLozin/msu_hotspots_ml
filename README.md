# Разработка методов детектирования пожаров по данным прибора МСУ-МР с российского спутника Метеор-М-2-2 с использованием методов машинного обучения
В настоящее время детектирование пожаров по данным спутникового мониторинга активно 
используется в информационных системах пожарного контроля (например, ИСДМ-Рослесхоз 
https://aviales.ru/default.aspx?textpage=117). Наиболее распространенным является детектирование 
пожаров по данным со спектрорадиометра MODIS, соответствующий алгоритм детектирования 
показывает наиболее устойчивые и надежные результаты. При этом известно, что эксплуатация 
спутников AQUA и TERRA, на которых установлен прибор MODIS, заканчивается в декабре 2025 года, 
что означает, во-первых, отсутствие источника данных для детектирования с 2026, а во-вторых, 
постепенную деградацию качества данных, получаемых в настоящее время. Альтернативным 
существующим решением является алгоритм детектирования, работающий по данным с прибора VIIRS, 
установленном на американском спутнике Suomi NPP. Данный алгоритм имеет некие отличия от 
алгоритма детектирования по MODIS, связанные со спецификой самих данных (250м разрешение VIIRS 
и 1 км MODIS), хотя так же показывает устойчивые результаты. В перспективе единственный источник 
данных для устойчивого детектирования пожаров на территории всей страны создает сильную 
зависимость от этого источника. Тем самым, задача детектирования пожаров по данным с других 
спутников имеет большую актуальность. Так же, учитывая современные реалии, существуют риски 
политического влияния на доступ к источникам данных. Поэтому в первую очередь, интерес 
представляет использование данных с прибора МСУ-МР, установленном на отечественном спутнике 
Метеор М 2-2. Для решения данной задачи предлагается использовать методы машинного обучения для 
создания модели, способной по информации с разных спектральных каналов МСУ-МР определять 
пожары.
В статье [1] изучается возможность применения методов машинного обучения для 
детектирования пожаров по спутниковым данным. В данной работе предложен подход, использующий 
данные сразу из нескольких источников (приборы C-SAR, MSI, SLSTR MODIS) с применением методов 
глубокого обучения. Так же приводится описание изменений стандартной модели U-Net, сделанных в 
рамках решаемой задачи, и предложена подходящая для задачи метрика. Весь этот материал может быть 
использован для нулевой итерации решения задачи детектирования горячих точек по данным МСУ-МР. 
Однако, следует учесть, что авторы данной работы уделили мало внимания технической применимости 
данной модели для потокового детектирования пожаров на больших территориях. Предложенная 
авторами модель получает на вход изображение размером 256 на 256 пикселей с размером пиксела 110 
на 130 метров. Таким образом, одно изображение покрывает площадь примерно в 3 квадратных 
километра. Площадь РФ составляет более 17 млн квадратных километров, что вызывает определенный 
вопросы в оперативности получаемых результатов, получаемых с использованием предложенной 
модели. Такое положение дел наталкивает на мысль о применении ML не к изображениям, а к 
табличным данным, полученным в результате обработки данных изображений. Преимуществом данного 
подхода является решение проблемы скорости работы модели, так как технология преобразования 
изображений, покрывающих большие территории (порядка 10 млн квадратных километров) в 
табличные данные уже отработана и не занимает много времени. Такой подход был реализован и его 
результаты описаны в нашей с соавторами работе [2]. В ней использовались контура пожаров, 
полученные в результате работы алгоритма MOD14 детектирования пожаров по данным MODIS. 
Данный алгоритм является достаточно надежным, чтобы получаемые с его применением контуры могли 
использоваться для формирования выборок по данным МСУ-МР. В работе использована модель 
Random Forest, которая после обучения показала хорошую полноту и плохую точность при обработке 
потоковых данных (99% и 50% соответственно). Таким образом, существует два принципиальных 
способа решения поставленной задачи с потенциальной возможностью их совместного применения – 
сегментация изображений глубокими сетями (модификация UNet) или классификация табличных 
данных со спектральными параметрами пикселей (Random Forest). Так же интерес представляет 
возможность комбинирования двух методов (локализация пожаров с помощью классификации 
табличных данных и создание их контуров с помощью сегментации).
Данные собираются самостоятельно. Имеется доступ к полному каталогу сцен МСУ-МР. Для 
решения задачи, как классификации табличных данных, набор данных уже был сформирован с 
использованием контуров пожаров по данным MODIS (см. [2]). На данный момент собранно порядка 
10000 записей положительных примеров (пиксель является пожарным) и порядка 100000 
отрицательных (не является пожаром). Данные репрезентативны, так как собирались по всем сценам за 
2022 год. В данных присутствует естественный дисбаланс, связанный с тем, что земли с пожаром 
намного меньше, чем без него. Всевозможные выбросы, связанные с возможностью обработки битых 
данных фильтруются отдельно. Для варианта решения задачи, как сегментации изображений 
предлагается создать набор данных, аналогично, как описано в работе [1]. В качестве опорных 
предлагается использовать все те же контуры пожаров по MODIS. При этом необходимо отдельно 
проработать локализацию пожаров, чтобы на вход модели не подавалось изображение размером 5000 на 
5000 пикселей в которой пожарными могут являться не более 1000 (чаще порядка всего нескольких 
десятков).
Наиболее подходящей метрикой для оценки качества работы модели является f1-score, которая 
отражает одновременно точность и полноту получаемого результата. Вместе с тем, оценки получаемых 
результатов при решении задач детектирования пожаров в разных работах сильно зависят от того, как 
проверять, что является пожаром, а что нет. Например, в работе [1], для самой лучшей комбинации 
источников данных f1-score достигает 87%. В работе [2] f1-score достигает 96.8%. Однако, первый 
результат базируются лишь на ограниченном наборе контуров пожаров, полученных по наземным 
данным, а второй вообще на ограниченном датасете, основанном на контурах пожаров, полученных в 
результате работы алгоритма MOD14 по данным с прибора MODIS. При этом в статье, оценивающей 
качество работы алгоритма MOD14, приводят цифры полноты (recall) в 68% и точности (accuracy) в 
85%. Такие низкие показатели связаны с тем, что в ней учитываются пожары, которые в принципе 
нельзя было увидеть со спутника, а также различные ложные детектирования, связанные с бликами 
солнца итп. Для данной работы целью является максимальное детектирование пожаров, которые 
визуально могут быть определены по всей сцене. Для модели из работы [2] при тестовой обработке 2 
дней сеансов f1-score упал до значения 67.3%. Поэтому ставится задача достижения f1-score>85% на 
всех обрабатываемых сеансах. То есть при получении высокого показателя f1-score (>95%), модель 
будет отдельно проверяться на потоковой обработке данных. При повторном снижении показателя, 
предлагается заново уточнять выборку данных.
Разработанную модель предлагается встроить в текущую обработку данных МСУ-МР, в рамках 
ЦКП-ИКИ Мониторинг (http://ckp.geosmis.ru/). Таким образом, данные со спутника будут доступны в 
полном объеме. Конечным пользователем являются службы охраны лесов, например, «Авиалесохрана». 
Модель предлагается реализовать на серверных мощностях 56 отдела ИКИ РАН. По результатам работы 
планируется публикация в журнале Современные проблемы дистанционного зондирования Земли из 
космоса (http://jr.rse.cosmos.ru/). Журнал индексируется в системах Scopus (CiteScore 1.8), Russian 
Science Citation Index (RSCI), РИНЦ, включен в перечень ВАК. 

Список литературы
1. D. Rashkovetsky, F. Mauracher, M. Langer, and M. Schmitt, “Wildfire detection from multisensor 
satellite imagery using deep semantic segmentation,” IEEE J. Sel. Topics Appl. Earth Observ. Remote 
Sens., vol. 14, pp. 7001–7016, Jun. 2021.
2. Д. В. Лозин, А. В. Кашницкий, Е. А. Лупян Разработка методов детектирования пожаров по 
данным МСУ-МР на основе обучающей выборки с использованием данных MODIS 
Материалы Десятой международной научно-технической конференции Актуальные проблемы 
создания космических систем дистанционнного зондирования земли 
АО «Корпорация «ВНИИЭМ» / 2022 / Москва / c 126-131

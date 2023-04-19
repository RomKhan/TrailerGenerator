# TrailerGenerator
Описание гиперпараметров для запуска через функцию main:
<ul>
<li>Параметр 1 - необходимость сжатия фильма. Значение 0 означает, что программа будет сжимать фильм и сохранять во временную папку. В случае значения 1 программа не будет сжимать фильм, но если сжатой версии на последующих шагах не окажется во временной папке, программа может выдать ошибку;</li>
<li>Параметр 2 - необходимость определения сцен. Значение 0 означает, что программа будет разграничивать фильма на сцены и сохранять файл с информацией о сценах во временную папку. В случае значения 1 программа не будет разграничивать фильма на сцены, но если файла с разграничением сцен на последующих шагах не окажется во временной папке, программа выдаст ошибку;</li>
<li>Параметр 3 - необходимость превращения данных о фильме в clip embedding’и. Значение 0 означает, что программа будет превращать все данные о фильме в json файле в clip embedding’и и сохранять файл с ними во временную папку. В случае значения 1 программа не будет ничего делать, но если файла с векторами признаков на последующих шагах не окажется во временной папке, программа выдаст ошибку;</li>
<li>Параметр 4 - количество сцен, которые должны оказаться в трейлере;</li>
<li>Параметр 5 - путь к фильму, для которого требуется сгенерировать трейлер;</li>
<li>Параметр 6 - путь к json файлу с информацией о фильме, содержащий следующие поля:</li>
<ul>
<li>ganre - массив с жанрами. Каждый жанр представлен строкой;</li>
<li>year - год выпуска фильма. Представлен целым натуральным числом;</li>
<li>Summaries - краткое описание фильма в текстовом виде. Представлен одной строкой;</li>
<li>synopsis - сюжет фильма в текстовом видео. Представлен одной строкой;</li>
<li>movie_threshold - значение порога для pyscenedetect модуля, который отвечает за резкость реагирования на изменения внутри сцены (чем меньше порог, тем более часто изменения внутри сцены приводят к делению сцены на 2). Представлен целым натуральным числом;</li>
<li>user (опционально) - пользовательский текст, который нужно учесть при формировании трейлера к фильму.</li>
</ul>
</ul>

В базовой вариации модель учитывает в равной степени Summaries, synopsis, ganre и user (0.5*(Summaries, synopsis, ganre) + 0.5 *user).</br>
Чтобы занулить коэффициент информации с (Summaries, synopsis, ganre) и учитывать только пользовательский запрос нужно в model.py в функции cosinus_dist_with_custom_description заменить коэффициент перед первым слагаемым в max_dist на 0

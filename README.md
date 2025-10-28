# NaCoSAE
Thesis Project - DS Bachelor Engineering Programme at WUT.

## Contriubutors
[Michał](https://github.com/piechotam) & [Gaspar](https://github.com/GasparSekula)

## Cel pracy

Technologia rzadkich autoenkoderów (Sparse Autoencoders, SAE) zyskała znakomitą popularność jako narzędzie służące do zwiększania interpretowalności modeli uczenia maszynowego określanych jako "black-box". Rosnące znaczenie tej architektury wynika z jej skutecznej adaptacji do wyjaśniania działania dużych modeli językowych (LLM) oraz z metody trenowania SAE w sposób samonadzorowany, który nie wymaga anotowanych danych. Rzadkie autokodery transformują wewnętrzną, polisemantyczną gęstą reprezentację danych w modelach na rzadką, bardziej monosemantyczną, a przez to bardziej zrozumiałą dla ludzi. Kluczowym wyzwaniem pozostaje jednak identyfikacja i automatyczne nazywanie monosemantycznych konceptów kodowanych przez poszczególne cechy (neurony) w SAE. O ile istnieją metody, które korzystają z modalności tekstowej, o tyle brakuje dedykowanych rozwiązań dla modeli opartych wyłącznie na modalności obrazowej. Głównym celem pracy jest opracowanie, implementacja i ocena nowej metody automatycznego nazywania cech (neuronów) w rzadkich autokoderach (SAE) trenowanych na modelach wizyjnych.  


## Tematyka pracy

Realizację celów niniejszej pracy inżynierskiej rozpocznie analiza literatury obejmującej przegląd metod interpretacji i nazywania cech w głębokich modelach wizyjnych oraz SAE (dla modalności tekstowej i multimodalnej), co stanowić będzie podstawę dla nowej metody i technik porównawczych. Następnie zespół opracuje nową metodykę automatycznego nazywania cech wizualnych w SAE. Etap ten będzie zawierał zapoznanie się i zaimplementowanie pętli optymalizacyjnej z modelem językowym (LLM) (w oparciu o framework MILS) i modelem generatywnym (np. Stable Diffusion) do tworzenia promptów, na bazie których generowane będą obrazki maksymalizujące aktywację cechy w SAE, a także wykorzystanie LLM do automatycznego proponowania nazw konceptów na podstawie tych promptów. 
Kolejnym krokiem będzie analiza i testowanie prototypowego systemu realizującego opracowaną metodykę. Przeprowadzona zostanie ewaluacja skuteczności nowej metody na modelach multimodalnych przy użyciu metryk z pracy CoSy oraz porównanie z istniejącymi podejściami. 
Następnie dla modelu o modalności tylko wizyjnej stworzona metoda zostanie zewaluowana. Pracę zwieńczy wizualizacja wyników, czyli opracowanie interfejsu użytkownika prezentującego dla wybranych cech SAE aktywujące prompty, wygenerowane obrazy, zaproponowane nazwy konceptów oraz wyniki ewaluacji.  


 ## Literatura
 - Ashutosh, K., Gandelsman, Y., Chen, X., Misra, I., and Girdhar, R. LLMs can see and hear without any training. arXiv preprint arXiv:2501.18096, 2025.
 - Zaigrajew, V., Baniecki, H., and Biecek, P. Interpreting CLIP with Hierarchical Sparse Autoencoders. arXiv preprint arXiv:2502.20578, 2025.
 - Rao, S., Mahajan, S., Böhle, M., and Schiele, B. Discover-then-Name: Task-Agnostic Concept Bottlenecks via Automated Concept Discovery. arXiv preprint arXiv:2407.14499, 2024.
 - Kopf, L., Bommer, P. L., Hedström, A., Lapuschkin, S., Höhne, M. M.-C., and Bykov, K. CoSy: Evaluating Textual Explanations of Neurons. arXiv preprint arXiv:2405.20331, 2024.

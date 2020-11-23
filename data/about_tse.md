# Note

You can get all stocks listed on Tokyo Stock Exchange (TSE) from [this url](https://www.jpx.co.jp/markets/statistics-equities/misc/01.html).

**「規模区分」について**

それぞれの記載は，以下の指数に組入れられていることを表す.

- TOPIX Core30 => TOPIX Core30, TOPIX 100, **TOPIX 500**, TOPIX 1000
- TOPIX Large70 => TOPIX Large70, TOPIX 100, **TOPIX 500**, TOPIX 1000
- TOPIX Mid400 => TOPIX Mid400, **TOPIX 500**, TOPIX 1000
- TOPIX Small 1 => TOPIX Small, TOPIX 1000
- TOPIX Small 2 => TOPIX Small

例: TOPIX500構成銘柄を集計したい場合，「規模区分」がTOPIX Core30, TOPIX Large70, TOPIX Mid400のいずれかである銘柄を集計すればよい．


```
■インデックスの主な構成　（2008年11月現在）

・TOPIX（東証株価指数）
・TOPIXニューインデックスシリーズ
　　　　TOPIX 1000
　　　　　　├TOPIX 500 （大型/中型株）
　　　　　　　　　├TOPIX 100 （大型株）
　　　　　　　　　　　　├TOPIX Core30 （超大型株）
　　　　　　　　　　　　└TOPIX Large70
　　　　　　　　　└TOPIX Mid400 （中型株）
　　　　　　└TOPIX Small （小型株）
・東証規模別株価指数（大型株・中型株・小型株）
・東証業種別株価指数（33業種）
・TOPIX-17シリーズ（33業種を17業種に集約）
・東証第二部株価指数
・東証マザーズ指数
```

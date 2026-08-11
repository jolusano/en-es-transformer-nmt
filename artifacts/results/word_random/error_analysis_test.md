### Worst 20 translations — en-es

| # | sBLEU | Source | Reference | Model output | Failure modes |
|---|-------|--------|-----------|--------------|---------------|
| 1 | 0.0 | Tell me about it! | Ni me lo digas. | ¡<unk> al respecto! | unknown_token, no_content_overlap |
| 2 | 0.0 | Curse you! | Te maldigo. | ¡<unk>! | unknown_token, no_content_overlap |
| 3 | 0.0 | Don't screw around with me. | ¡No jodas conmigo! | No <unk> la <unk>. | repetition, over_generation, unknown_token, no_content_overlap |
| 4 | 0.0 | Tell me about it. | ¡Dímelo a mí! | Háblame al respecto. | no_content_overlap |
| 5 | 2.8 | Dial 411 for directory assistance. | Marque 411 para asistencia de directorio. | No <unk> la <unk> por ayuda <unk>. | repetition, over_generation, unknown_token, number_mismatch, no_content_overlap |
| 6 | 3.4 | Ecosystems are degrading rapidly all around the Earth. | Los ecosistemas se están degradando en todo el mundo. | Los <unk> son <unk> rápidamente por toda la Tierra. | repetition, unknown_token, no_content_overlap |
| 7 | 3.4 | Even the hard-hearted can be moved to tears. | Hasta el diablo llora. | Incluso los <unk> se pueden mover a las lágrimas. | over_generation, unknown_token, no_content_overlap |
| 8 | 3.6 | His meaning is quite plain. | Lo que él trata de decir es bien sencillo de comprender. | Su significado es bastante simple. | truncation, no_content_overlap |
| 9 | 3.7 | It's coming out! My bladder's about to explode! | ¡Me estoy meando, la vejiga me va a explotar! | ¡Ya sale! ¡Mi <unk> está a punto de estallar! | unknown_token, no_content_overlap |
| 10 | 3.7 | The Geneva Convention is not recognized at Guantanamo. | La Convención de Ginebra no está reconocida en Guantanamo. | La convención de <unk> no se reconoce a las <unk>. | repetition, unknown_token |
| 11 | 3.7 | Crap! My girlfriend is already back! What should we do?! | ¡Mierda, mi polola ya volvió! ¿Qué hacemos? | ¡demonios! ¡Mi novia ya está de vuelta! ¡¿Qué deberíamos hacer?! | other |
| 12 | 3.9 | Both Tom and John married Canadian women. | Tom y John están, los dos, casados con mujeres canadienses. | Tanto Tom como John se casaron canadiense. | other |
| 13 | 3.9 | And passing on from thence to a mountain that was on the east side of Bethel, he there pitched his tent, having Bethel on the west and Hai on the east: he built there also an altar to the Lord, and called upon his name. | De allí pasó a la montaña, al oriente de Betel, y desplegó su tienda, entre Betel al occidente y Ay al oriente. Allí edificó un altar a Yahvé e invocó su nombre. | Y pasar por <unk> hasta una montaña que estaba en el lado del <unk>, <unk> su carpa, teniendo <unk> en el oeste y <unk> en el este: él construyó también un altar hacia el Señor y llamó a su nombre. | repetition, unknown_token |
| 14 | 4.0 | Scholarships are awarded to undeserving people and good students are denied. | Se conceden becas a personas que no las merecen, y se niegan a buenos estudiantes. | Los <unk> son <unk> para <unk> a la gente y los buenos estudiantes son <unk>. | repetition, unknown_token |
| 15 | 4.0 | We make a teeny bit of progress, then we go back to square one. | Apenas logramos un avance, todo vuelve a cero. | <unk> un poco de progreso, entonces volvemos a la casilla. | unknown_token, no_content_overlap |
| 16 | 4.0 | Cardinals are, in general, excellent polyglots. | Los Cardenales son por lo general excelentes Políglotas. | Los <unk> son, en general, excelente <unk>. | repetition, unknown_token |
| 17 | 4.0 | Watch him and do likewise. | Vigílale y haz lo mismo. | <unk> y <unk> las <unk>. | repetition, over_generation, unknown_token, no_content_overlap |
| 18 | 4.0 | The stevia was cultivated and used as a sweetener by the Guarani people for at least 1,500 years. | La stevia fue cultivada y utilizada como edulcorante por el pueblo guaraní durante al menos 1.500 años. | Los <unk> fueron <unk> y <unk> como <unk> por la gente <unk> por al menos 1. 500 años. | repetition, unknown_token |
| 19 | 4.2 | He has a clumsy walk. | Camina de forma torpe. | Él tiene un <unk> <unk>. | repetition, over_generation, unknown_token, no_content_overlap |
| 20 | 4.2 | Here's a restaurant I eat at a lot. | Aquí hay un restaurante en el que como a menudo. | Aquí tengo mucho restaurante. | truncation |

**Failure-mode frequency across the whole test set**

| Category | Sentences | Rate |
|----------|-----------|------|
| other | 4362 | 77.1% |
| unknown_token | 1119 | 19.8% |
| no_content_overlap | 243 | 4.3% |
| repetition | 208 | 3.7% |
| over_generation | 106 | 1.9% |
| number_mismatch | 14 | 0.2% |
| truncation | 12 | 0.2% |
| copied_source | 8 | 0.1% |


### Worst 20 translations — es-en

| # | sBLEU | Source | Reference | Model output | Failure modes |
|---|-------|--------|-----------|--------------|---------------|
| 1 | 0.0 | Te maldigo. | Curse you! | You recovered. | no_content_overlap |
| 2 | 3.1 | Los arpones de Philae no se desplegaron. | Philae's harpoons did not deploy. | The <unk> of <unk> will not be <unk>. | repetition, over_generation, unknown_token, no_content_overlap |
| 3 | 3.2 | Considera a las mujeres placeres de usar y tirar y no búsquedas con sentido. | He regards women as disposable pleasures rather than as meaningful pursuits. | Consider the women pleasures to use and flush and not to be <unk>. | unknown_token |
| 4 | 3.2 | Estados Unidos se imagina que es la nación más libre del mundo. | America fancies itself the world's freest nation. | The United States is imagining that it's the <unk> nation in the world. | over_generation, unknown_token |
| 5 | 3.5 | Los solsticios y los equinoccios marcan el inicio de las estaciones del año. | Solstices and equinoxes mark the beginnings of the seasons of the year. | The <unk> and the <unk> begin the <unk> stations. | repetition, unknown_token, no_content_overlap |
| 6 | 3.7 | Considera que las mujeres son placeres de usar y tirar más que búsquedas con sentido. | He regards women as disposable pleasures rather than as meaningful pursuits. | Consider that women are pleasures to use and pull more than <unk>. | unknown_token |
| 7 | 3.7 | Considera que las mujeres son placeres desechables más que objetivos significativos. | He regards women as disposable pleasures rather than as meaningful pursuits. | Consider that women are <unk> pleasures more than <unk> goals. | repetition, unknown_token |
| 8 | 4.0 | Considera a las mujeres placeres desechables y no propósitos con sentido. | He regards women as disposable pleasures rather than as meaningful pursuits. | Consider the Turkish women with no purpose. | other |
| 9 | 4.0 | Marque 411 para asistencia de directorio. | Dial 411 for directory assistance. | outline <unk> to care for this <unk>. | repetition, over_generation, unknown_token, number_mismatch, no_content_overlap |
| 10 | 4.0 | ¡Me estoy meando, la vejiga me va a explotar! | It's coming out! My bladder's about to explode! | I'm <unk>, bladder is going to blow up! | unknown_token, no_content_overlap |
| 11 | 4.0 | Las lombrices a veces son beneficiosas para el suelo. | Worms can be good for the soil. | <unk> are sometimes <unk> to the ground. | repetition, unknown_token, no_content_overlap |
| 12 | 4.0 | Tenga cuidado de no resbalar en las baldosas mojadas. | Mind you don't slip on the wet tiles. | Be careful not to sneak in your wet <unk>. | unknown_token |
| 13 | 4.1 | Las flores de loto desprendían una fragancia indescriptible. | The lotus flowers gave off an indescribable fragrance. | <unk> flowers <unk> an <unk> scent. | repetition, unknown_token |
| 14 | 4.1 | Él pasó su cepillo por toda la lona. | He swept his brush across the canvas. | He spent his <unk> all over his <unk>. | repetition, unknown_token |
| 15 | 4.1 | Un sólido argumento en favor del cristianismo es el siguiente: las ofensas contra la caridad es probablemente lo único que, en sus lechos de muerte, los hombres llegan a sentir y no a comprender como un crimen. | A strong argument for the religion of Christ is this - that offences against Charity are about the only ones which men on their death-beds can be made, not to understand, but to feel, as crime. | An argument solid in favor of Christianity is the following: the trespasses against charity is probably the only thing that, in his <unk> of death, men came to feel and don't understand a crime. | unknown_token |
| 16 | 4.2 | Me parece que el tiempo se está despejando. | I think it's clearing up. | It seems to me that time is getting tired. | no_content_overlap |
| 17 | 4.3 | Esto viola alrededor de un millón de códigos penales y todos los libros sagrados existentes. | This violates about a million penal codes and every holy book there is. | This viola around a million <unk> <unk> and all the <unk> <unk> books. | repetition, unknown_token |
| 18 | 4.4 | Las calles adoquinadas le dan un aspecto muy pintoresco a este pueblo. | The cobblestone streets give this town a very picturesque appearance. | <unk> streets bring this village very <unk> looks. | repetition, unknown_token |
| 19 | 4.5 | Placer y alegría tan pronto ida como venida. | Happiness comes and goes. | Pleasure and joy as soon as I <unk>. | over_generation, unknown_token, no_content_overlap |
| 20 | 4.5 | No lo voy a soportar más. | I won't tolerate it anymore. | I'm not going to put up with it any longer. | over_generation, no_content_overlap |

**Failure-mode frequency across the whole test set**

| Category | Sentences | Rate |
|----------|-----------|------|
| other | 4781 | 84.5% |
| unknown_token | 701 | 12.4% |
| no_content_overlap | 167 | 3.0% |
| repetition | 138 | 2.4% |
| over_generation | 45 | 0.8% |
| number_mismatch | 25 | 0.4% |
| truncation | 11 | 0.2% |
| copied_source | 4 | 0.1% |

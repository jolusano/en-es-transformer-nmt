### Worst 20 translations — en-es

| # | sBLEU | Source | Reference | Model output | Failure modes |
|---|-------|--------|-----------|--------------|---------------|
| 1 | 0.0 | Everyone escaped. | Escaparon todos. | Todos escaparon. | other |
| 2 | 0.0 | Thank you just the same. | Gracias de todos modos. | Gracias solo. | other |
| 3 | 0.0 | Slow down! | ¡Despacio! | ¡Órale! | no_content_overlap |
| 4 | 0.0 | See you later. | Nos estamos viendo. | Hasta luego. | no_content_overlap |
| 5 | 0.0 | I am Japanese. | Soy japonesa. | Soy japonés. | other |
| 6 | 0.0 | I hate it. | Me la seca. | Lo odio. | no_content_overlap |
| 7 | 0.0 | I followed him. | Le seguí. | Lo seguí. | other |
| 8 | 0.0 | We're credible. | Somos confiables. | Somos creíble. | other |
| 9 | 0.0 | I belched. | Eructé. | Me pegué. | no_content_overlap |
| 10 | 0.0 | When do we go? | ¿Cuándo vamos? | ¿Cuándo vamos? | other |
| 11 | 0.0 | Does she like me? | ¿Le gusto a ella? | ¿Le gusto? | other |
| 12 | 0.0 | Who do you love? | ¿A quién ama usted? | ¿Quién amas? | no_content_overlap |
| 13 | 0.0 | It happens to me. | A mí me sucede. | Me pasa. | other |
| 14 | 0.0 | I am Japanese. | Soy japonés. | Soy japonés. | other |
| 15 | 0.0 | He will study French. | Él estudiará francés. | Estudiará francés. | other |
| 16 | 0.0 | Surprise me. | Sorprendedme. | Caredadme. | no_content_overlap |
| 17 | 0.0 | I learn fast. | Aprendo rápido. | Aprendo rápido. | other |
| 18 | 0.0 | I want to come in. | Quiero entrar. | Quiero entrar. | other |
| 19 | 0.0 | She is quiet. | Ella es tranquila. | Está tranquila. | other |
| 20 | 0.0 | I have gained weight. | He ganado peso. | He engordado. | no_content_overlap |

**Failure-mode frequency across the whole test set**

| Category | Sentences | Rate |
|----------|-----------|------|
| other | 5313 | 93.9% |
| no_content_overlap | 260 | 4.6% |
| truncation | 41 | 0.7% |
| repetition | 29 | 0.5% |
| over_generation | 16 | 0.3% |
| number_mismatch | 10 | 0.2% |
| copied_source | 7 | 0.1% |


### Worst 20 translations — es-en

| # | sBLEU | Source | Reference | Model output | Failure modes |
|---|-------|--------|-----------|--------------|---------------|
| 1 | 0.0 | Gracias de todos modos. | Thank you just the same. | Thanks anyway. | truncation, no_content_overlap |
| 2 | 0.0 | ¡Despacio! | Slow down! | Get lost! | no_content_overlap |
| 3 | 0.0 | Voy a afeitarme. | I'm going to shave. | I'll shave. | other |
| 4 | 0.0 | Somos confiables. | We're credible. | We're trust. | other |
| 5 | 0.0 | Eructé. | I belched. | I burped. | no_content_overlap |
| 6 | 0.0 | Los políticos mienten. | Politicians lie. | Politicians lie. | other |
| 7 | 0.0 | Gracias de todas formas. | Thanks all the same. | Thanks anyway. | other |
| 8 | 0.0 | Sorprendedme. | Surprise me. | Answer me. | other |
| 9 | 0.0 | Ven a casa. | Come home. | Come home. | other |
| 10 | 0.0 | Tomás esperó. | Tom waited. | Tom waited. | other |
| 11 | 0.0 | No hay vuelta atrás. | There is no going back. | Don't back. | truncation |
| 12 | 0.0 | Ahora es julio. | It's July. | It's July. | other |
| 13 | 0.0 | Estupendo. | I stayed up late. | I stay. | no_content_overlap |
| 14 | 0.0 | Elige sabiamente. | Choose carefully. | Choose wisely. | other |
| 15 | 0.0 | Vitorearon. | They cheered. | They acted. | no_content_overlap |
| 16 | 0.0 | ¿Quién se está riendo? | Who's laughing? | Who's laughing? | other |
| 17 | 0.0 | Siéntense allá. | Sit over there. | Sit there. | other |
| 18 | 0.0 | Gracias de todas formas. | Thanks anyway. | Thanks anyway. | other |
| 19 | 0.0 | Bésame. | Kiss me. | Kiss me. | other |
| 20 | 0.0 | ¡Sentaos! | Sit down! | Stay! | no_content_overlap |

**Failure-mode frequency across the whole test set**

| Category | Sentences | Rate |
|----------|-----------|------|
| other | 5349 | 94.6% |
| no_content_overlap | 221 | 3.9% |
| repetition | 35 | 0.6% |
| truncation | 33 | 0.6% |
| number_mismatch | 19 | 0.3% |
| over_generation | 9 | 0.2% |
| copied_source | 5 | 0.1% |

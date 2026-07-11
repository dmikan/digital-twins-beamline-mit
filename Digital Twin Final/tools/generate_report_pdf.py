import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        
        # We suppress headers and footers on the cover page (Page 1)
        if self._pageNumber > 1:
            # Color palette
            charcoal = colors.HexColor('#2D3748')
            slate_gray = colors.HexColor('#718096')
            border_color = colors.HexColor('#E2E8F0')
            
            # --- RUNNING HEADER ---
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor('#1A365D'))
            self.drawString(54, 750, "DISEÑO Y OPTIMIZACIÓN DE LÍNEA DE HAZ DE IONES")
            
            self.setFont("Helvetica", 8)
            self.setFillColor(slate_gray)
            self.drawRightString(558, 750, "Informe Técnico | Gemelo Digital v2")
            
            # Header separator line
            self.setStrokeColor(border_color)
            self.setLineWidth(0.75)
            self.line(54, 742, 558, 742)
            
            # --- RUNNING FOOTER ---
            # Footer separator line
            self.line(54, 52, 558, 52)
            
            self.setFont("Helvetica", 8)
            self.setFillColor(slate_gray)
            self.drawString(54, 40, "Confidencial - Trabajo de Investigación")
            
            page_text = f"Página {self._pageNumber} de {page_count}"
            self.drawRightString(558, 40, page_text)
            
        self.restoreState()

def add_image_or_placeholder(story, img_path, width, height, caption, caption_style):
    import os
    import struct
    from reportlab.platypus import Paragraph, Image, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    
    if os.path.exists(img_path):
        img_w, img_h = width, height
        try:
            with open(img_path, 'rb') as f:
                data = f.read(24)
                if data[:8] == b'\x89PNG\r\n\x1a\n' and data[12:16] == b'IHDR':
                    w, h = struct.unpack('>ii', data[16:24])
                    # Compute correct height based on the PDF width
                    img_h = int((h / w) * width)
                    img_w = width
        except Exception:
            pass
        story.append(Image(img_path, width=img_w, height=img_h))
    else:
        # Create a beautiful placeholder box
        data = [[Paragraph(f"<font color='#718096'><b>[ESPACIO PARA FIGURA EN PDF]</b><br/>"
                           f"Archivo no encontrado: <i>{os.path.basename(img_path)}</i><br/>"
                           f"Usa este espacio para montar la figura de forma manual.</font>", 
                           ParagraphStyle('PlaceholderText', parent=caption_style, alignment=1, fontSize=9, leading=13))]]
        t = Table(data, colWidths=[width], rowHeights=[height])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F7FAFC')),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#E2E8F0')),
            ('PADDING', (0,0), (-1,-1), 12),
        ]))
        story.append(t)
    story.append(Paragraph(caption, caption_style))

def create_report():
    pdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../outputs/Informe_Parcial.pdf'))
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    
    # Page setup
    # Letter size: 612 x 792 pt
    # Margins: 54 pt (0.75 inch) -> usable width: 504 pt (17.78 cm)
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=72,  # Give room for running header
        bottomMargin=72  # Give room for running footer
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Color Palette
    primary_color = colors.HexColor('#1A365D')    # Deep Navy
    secondary_color = colors.HexColor('#2B6CB0')  # Slate Blue
    accent_color = colors.HexColor('#D69E2E')     # Muted Gold
    text_color = colors.HexColor('#2D3748')       # Charcoal
    bg_light = colors.HexColor('#F7FAFC')         # Warm White/Light Gray
    border_color = colors.HexColor('#E2E8F0')     # Border Gray
    
    # Custom Typography Styles
    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=30,
        textColor=primary_color,
        spaceAfter=10
    )
    
    subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=13,
        leading=18,
        textColor=secondary_color,
        spaceAfter=30
    )
    
    meta_label_style = ParagraphStyle(
        'MetaLabel',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=primary_color
    )
    
    meta_val_style = ParagraphStyle(
        'MetaValue',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=text_color
    )
    
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=primary_color,
        spaceBefore=15,
        spaceAfter=10,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=secondary_color,
        spaceBefore=12,
        spaceAfter=8,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14.5,
        textColor=text_color,
        spaceAfter=10
    )
    
    bullet_style = ParagraphStyle(
        'ReportBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14.5,
        textColor=text_color,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=5
    )
    
    callout_text_style = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#4A5568')
    )
    
    caption_style = ParagraphStyle(
        'ImageCaption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        alignment=1, # Centered
        textColor=colors.HexColor('#718096'),
        spaceBefore=6,
        spaceAfter=15
    )
    
    story = []
    
    # ---------------------------------------------------------
    # COVER PAGE
    # ---------------------------------------------------------
    story.append(Spacer(1, 4 * cm))
    
    # Decorative line block
    d_line = Table([[""]], colWidths=[504], rowHeights=[4])
    d_line.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), primary_color),
        ('PADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(d_line)
    story.append(Spacer(1, 0.5 * cm))
    
    story.append(Paragraph("Optimización de Beamline de Iones mediante un Gemelo Digital Físico", title_style))
    story.append(Paragraph("Estructura y Justificación de la Cantidad a Modelar e Implementación del Filtro Físico", subtitle_style))
    
    story.append(Spacer(1, 1 * cm))
    
    # Metadata Box
    meta_data = [
        [Paragraph("Proyecto:", meta_label_style), Paragraph("Digital Twins for Beamline (MIT Collaboration)", meta_val_style)],
        [Paragraph("Autores:", meta_label_style), Paragraph("Julian Andres Suarez Cardona (julsuarezca@unal.edu.co), Pablo José Montoya Barraza (pmontoyab@unal.edu.co), Danny Julián Perilla Mikán (djperillam@unal.edu.co).", meta_val_style)],
        [Paragraph("Fecha del Reporte:", meta_label_style), Paragraph("10 de julio de 2026", meta_val_style)],
        [Paragraph("Entorno de Simulación:", meta_label_style), Paragraph("SIMION 8.1 & Custom Runge-Kutta 4 (RK4) Vectorizado", meta_val_style)],
    ]
    meta_table = Table(meta_data, colWidths=[120, 384])
    meta_table.setStyle(TableStyle([
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(meta_table)
    
    story.append(Spacer(1, 4 * cm))
    
    # Bottom accent
    accent_bar = Table([["", ""]], colWidths=[20, 484], rowHeights=[6])
    accent_bar.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), accent_color),
        ('BACKGROUND', (1,0), (1,0), secondary_color),
        ('PADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(accent_bar)
    
    story.append(PageBreak())
    
    # ---------------------------------------------------------
    # SECTION 1: LA CANTIDAD ELEGIDA PARA MODELAR Y POR QUÉ
    # ---------------------------------------------------------
    story.append(Paragraph("1. La Cantidad Elegida para Modelar y Justificación Física", h1_style))
    
    p1_1 = (
        "El objetivo del gemelo digital no consiste en modelar directamente el campo eléctrico del sistema "
        "ni las trayectorias individuales de las partículas, sino en aproximar una función objetivo escalar que "
        "representa la calidad global del haz obtenido para una configuración determinada de voltajes. Esta cantidad "
        "corresponde a la función <b>J<sub>v2.4</sub></b>, la cual integra en un único valor las características "
        "más relevantes del haz y permite comparar objetivamente distintas configuraciones de operación."
    )
    story.append(Paragraph(p1_1, body_style))
    
    story.append(Paragraph("Ventajas de una única función objetivo", h2_style))
    p1_2 = (
        "La elección de una única función objetivo presenta varias ventajas:<br/>"
        "• <b>Reducción de la dimensionalidad:</b> Reduce el problema de optimización a minimizar un escalar en lugar de optimizar simultáneamente múltiples variables físicas.<br/>"
        "• <b>Eficiencia computacional:</b> La evaluación de esta función puede realizarse mediante el modelo RK4 del gemelo digital, evitando ejecutar una simulación completa en SIMION para cada candidato y reduciendo considerablemente el costo computacional."
    )
    story.append(Paragraph(p1_2, body_style))
    
    story.append(Paragraph("Métricas e indicadores del haz", h2_style))
    p1_3 = (
        "La función objetivo se calcula a partir de las características extraídas del haz mediante el módulo <code>caracterizador.py</code>, el cual analiza las trayectorias finales de todas las partículas y calcula métricas como:<br/>"
        "• Número de partículas que alcanzan el detector (hits).<br/>"
        "• Distancia media de las partículas al detector.<br/>"
        "• <i>Offset</i> del centro del haz.<br/>"
        "• Tamaño del haz (&sigma;<sub>x</sub> y &sigma;<sub>y</sub>).<br/>"
        "• Divergencia angular.<br/>"
        "• Parámetros de Twiss (&alpha;, &beta;, &gamma;).<br/>"
        "• Emitancias.<br/>"
        "• Fracción de halo.<br/>"
        "• Kurtosis.<br/>"
        "• Residuos del transporte (matriz de transferencia)."
    )
    story.append(Paragraph(p1_3, body_style))
    
    p1_4 = (
        "Cada una de estas cantidades representa un aspecto diferente de la calidad del haz. Posteriormente se combinan mediante una suma ponderada que produce el valor final de J<sub>v2.4</sub>."
    )
    story.append(Paragraph(p1_4, body_style))
    
    story.append(Paragraph("Evolución hacia la transmisión lineal (J_v2.4)", h2_style))
    p1_5 = (
        "La versión final del trabajo utiliza la función J<sub>v2.4</sub>, que incorpora una modificación importante respecto a versiones anteriores. Inicialmente la transmisión del haz se evaluaba mediante un término saturante, el cual dejaba de distinguir adecuadamente configuraciones con un número elevado de impactos sobre el detector. Para solucionar este inconveniente se añadió un término lineal de transmisión:<br/><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>&lambda; * (1 - hits / N<sub>ref</sub>)</b><br/><br/>"
        "utilizando <b>&lambda; = 2</b> y N<sub>ref</sub> = 500 (el total de iones). Este término aumenta la importancia del número de partículas detectadas durante la optimización y permite diferenciar correctamente soluciones con diferencias relativamente pequeñas en la transmisión, sin alterar el refinamiento local proporcionado por el resto de métricas."
    )
    story.append(Paragraph(p1_5, body_style))
    
    p1_6 = (
        "Finalmente, la función objetivo se normaliza para mantener aproximadamente el rango [0,1], conservando la interpretación de que valores menores corresponden a mejores configuraciones del sistema."
    )
    story.append(Paragraph(p1_6, body_style))
    
    story.append(PageBreak())
    
    # ---------------------------------------------------------
    # SECTION 2: RECOLECCIÓN DE DATOS Y COBERTURA DEL ESPACIO
    # ---------------------------------------------------------
    story.append(Paragraph("2. Recolección de Datos y Cobertura del Espacio de Búsqueda", h1_style))
    
    p2_1 = (
        "Los datos utilizados para entrenar el modelo fueron obtenidos mediante un esquema híbrido que combina simulaciones físicas rápidas con simulaciones de alta fidelidad."
    )
    story.append(Paragraph(p2_1, body_style))
    
    p2_2 = (
        "Cada candidato corresponde a una configuración de voltajes aplicada sobre los electrodos libres del sistema. Inicialmente dicha configuración es evaluada mediante el modelo RK4 implementado dentro del gemelo digital, el cual integra las ecuaciones de movimiento utilizando los campos eléctricos previamente calculados mediante superposición de funciones base. Esta evaluación proporciona una estimación rápida de la calidad del haz y permite descartar configuraciones claramente desfavorables."
    )
    story.append(Paragraph(p2_2, body_style))
    
    p2_3 = (
        "Las configuraciones más prometedoras son posteriormente evaluadas mediante SIMION, que constituye el modelo físico de referencia. Los resultados obtenidos se utilizan tanto para validar el gemelo digital como para continuar alimentando el proceso de optimización."
    )
    story.append(Paragraph(p2_3, body_style))
    
    p2_4 = (
        "Todas las evaluaciones quedan almacenadas automáticamente mediante <b>Optuna</b>, registrando para cada ensayo:<br/>"
        "• Voltajes aplicados.<br/>"
        "• Valor de la función objetivo.<br/>"
        "• Número de impactos sobre el detector (hits).<br/>"
        "• Características físicas del haz.<br/>"
        "• Información auxiliar utilizada posteriormente por el Proceso Gaussiano."
    )
    story.append(Paragraph(p2_4, body_style))
    
    story.append(Paragraph("Reparametrización Física del Cuadrupolo (Bender)", h2_style))
    p2_5 = (
        "En la versión final del trabajo se empleó una reparametrización del cuadrupolo, reduciendo el número de variables independientes del problema. En lugar de optimizar directamente los cuatro voltajes de los electrodos del cuadrupolo (V<sub>9</sub> a V<sub>12</sub>), éstos se describen mediante tres parámetros físicos:<br/>"
        "• <b>A:</b> desplazamiento común (offset) del cuadrupolo.<br/>"
        "• <b>B:</b> intensidad principal de flexión.<br/>"
        "• <b>C:</b> término de asimetría horizontal."
    )
    story.append(Paragraph(p2_5, body_style))
    
    p2_6 = (
        "Los voltajes físicos de los electrodos del bender se obtienen a partir de estos parámetros mediante las relaciones lineales:<br/><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>V<sub>9</sub> = A + B + C</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>V<sub>10</sub> = A - B</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>V<sub>11</sub> = A - B</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>V<sub>12</sub> = A + B - C</b><br/><br/>"
        "Esta parametrización reduce la dimensionalidad efectiva del problema sin perder la estructura física del sistema, facilitando el aprendizaje del modelo probabilístico."
    )
    story.append(Paragraph(p2_6, body_style))
    
    story.append(Spacer(1, 0.4 * cm))
    
    # Cobertura del Espacio Image (Figura 2.1)
    img_path_2 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../outputs/report_figures/fig_cobertura_espacio.png'))
    add_image_or_placeholder(story, img_path_2, 380, 308, 
                             "Figura 2.1: Cobertura del espacio de búsqueda en los electrodos del bender (V9 vs V12). Los puntos de color claro representan evaluaciones con 0 hits (muriendo en las paredes), mientras que los puntos sólidos de colores indican configuraciones exitosas con transmisión (hits > 0) para cada campaña de optimización. Se observa cómo el modelo Bender 6D se confina exactamente a la diagonal simétrica cuadrupolar (V9 = V12), mientras que las demás campañas exploran la cuenca asimétrica.", caption_style)
    
    story.append(PageBreak())
    
    # ---------------------------------------------------------
    # SECTION 3: EL MODELO: ARQUITECTURA, HIPERPARÁMETROS Y VALIDACIÓN
    # ---------------------------------------------------------
    story.append(Paragraph("3. El Modelo: Arquitectura, Hiperparámetros y Validación", h1_style))
    
    story.append(Paragraph("3.1 Arquitectura General", h2_style))
    p3_1_1 = (
        "El diseño del gemelo digital se fundamenta en estructurar un modelo de simulación alternativo que actúa como "
        "filtro de bajo costo computacional frente a los análisis detallados en el entorno de SIMION. Para simplificar "
        "el control de este flujo, el sistema consolida todas sus operaciones en la clase fachada <code>GemeloDigital</code> "
        "en <code>gemelo.py</code>, la cual actúa como punto de acceso central para coordinar el motor de física, el "
        "caracterizador del haz de iones y el lazo de optimización."
    )
    story.append(Paragraph(p3_1_1, body_style))
    
    p3_1_2 = (
        "La estrategia de búsqueda opera mediante un lazo de <i>screening</i> en dos etapas secuenciales:<br/>"
        "1. <b>Primera Etapa (Económica):</b> El módulo de optimización en <code>optimizer.py</code> genera un lote amplio de configuraciones candidatas (habitualmente 239 conjuntos de voltajes) que se envían al motor de física de <code>physics.py</code>. Este motor propaga las partículas en paralelo y estima la viabilidad geométrica del haz de cada candidato.<br/>"
        "2. <b>Segunda Etapa (De Alta Fidelidad):</b> El sistema selecciona las configuraciones más prometedoras (el top 10 de la evaluación anterior) y las ejecuta en el entorno real de SIMION. Los datos de impactos (<i>hits</i>) y perfiles obtenidos en SIMION se devuelven al optimizador para actualizar el Proceso Gaussiano (GP) que orienta las sugerencias de la siguiente iteración del lazo."
    )
    story.append(Paragraph(p3_1_2, body_style))
    
    p3_1_3 = (
        "<b>Justificación del costo de ejecución:</b> Cada corrida en SIMION consume cerca de 5.77 s de tiempo de pared, retraso provocado principalmente por el arranque de <code>fastadj</code> y la lectura en disco del archivo de potenciales de 290 MB [C1]. Dado que la zona útil de voltajes que transmite partículas representa una fracción extremadamente delgada del espacio de búsqueda de ocho dimensiones —una parte en mil millones (1 × 10<sup>-9</sup>) [C2]—, una búsqueda directa sobre SIMION consumiría semanas de cálculo. El motor de física intermedio resuelve esta limitación al evaluar cada candidato en apenas 0.25 s, una relación de velocidad de ~23x que permite filtrar lotes masivos de 239 candidatos en el mismo tiempo de pared en que SIMION resolvería solo 10 [C1]."
    )
    story.append(Paragraph(p3_1_3, body_style))
    
    story.append(Paragraph("3.2 El Motor de Física", h2_style))
    p3_2_1 = (
        "El motor de física modela la trayectoria del haz de iones resolviendo numéricamente las fuerzas electrostáticas locales en cada paso temporal. La simulación se organiza en cuatro capas de implementación:<br/>"
        "• <b>Superposición de potenciales:</b> En lugar de calcular la física del campo para cada candidato desde cero, se explota el carácter lineal de la ecuación de Laplace: el campo eléctrico resultante de aplicar cualquier combinación de voltajes se obtiene mediante la combinación lineal de 19 campos base de los electrodos, extraídos de SIMION una sola vez. En el módulo <code>physics.py</code>, estos campos se combinan mediante un producto tensorial matricial directo provisto por <code>numpy</code>. El campo de fuerza local se obtiene resolviendo los gradientes tridimensionales negativos sobre la grilla de potenciales combinada.<br/>"
        "• <b>Enrutamiento de grillas compuestas:</b> Para no saturar la memoria con un mapa uniforme de alta resolución, se estructura una grilla compuesta en la clase <code>CampoFino</code> de <code>physics.py</code>. Esta grilla superpone cajas locales de alta precisión (el cuadrupolo a 1.0 mm de espaciado, y los canales de colimación c1 y c2 a 2.5 mm de paso) sobre una grilla global más gruesa de 2.0 mm. La consulta de posiciones utiliza un esquema de prioridad en el que cada coordenada es resuelta por la caja fina más específica que la contiene, y solo los puntos residuales caen al mapa global. Esta técnica reduce la huella de memoria en unas 15x en comparación con un mapa homogéneo de 1.0 mm, manteniendo la misma precisión y correlación de ranking de salida. La interpolación espacial continua se vectoriza mediante <code>scipy.ndimage</code>, evitando bucles en Python.<br/>"
        "• <b>Integración de trayectorias:</b> La física de la trayectoria se propaga en un solo paso matricial para todo el lote de partículas y candidatos, mediante el método numérico de Runge-Kutta de cuarto orden (RK4) implementado en la clase <code>BatchRK4Integrator</code> de <code>physics.py</code>. Las ecuaciones de actualización integran, en cada paso temporal, la posición y la velocidad de los iones a partir de las fuerzas electrostáticas locales. La aceleración de los iones se obtiene a partir de su carga, su masa y el campo eléctrico interpolado, multiplicada por un factor de escala constante para convertir las unidades de campo (V/mm) y masa a unidades físicamente coherentes de mm/s<sup>2</sup>. Con un paso temporal de 1 × 10<sup>-8</sup> s, el avance espacial de las partículas es de aproximadamente 0.6 mm, lo que permite resolver la trayectoria en <i>gaps</i> estrechos de forma sub-milimétrica.<br/>"
        "• <b>Geometría de colisiones:</b> Para determinar qué partículas chocan contra las paredes sólidas, se extrae una máscara metálica directamente de la grilla de potenciales del archivo <code>.PA0</code> de SIMION (donde los puntos metálicos activos se definen por potencial y tipo de electrodo), lo que elimina cualquier desfase geométrico. En el código de <code>ParedesPA</code> de <code>physics.py</code>, estos vóxeles metálicos se estructuran en un árbol espacial para búsquedas rápidas de vecinos más cercanos. Durante la integración, el módulo consulta la distancia de cada ion a la pared más cercana. Para minimizar el costo de cómputo, el chequeo de colisiones se ejecuta cada 3 pasos de integración, logrando una reducción de costo de ~4.2x con la misma precisión en la clasificación de supervivencia. Si la distancia cae por debajo de un margen límite de contacto de 0.5 mm, la partícula se clasifica como colisionada.<br/>"
        "• <b>Caracterización del haz y función objetivo:</b> El caracterizador en <code>caracterizador.py</code> mide las propiedades geométricas del haz (distribución espacial, divergencia, emitancia Twiss) en el plano del detector (z = 390.0 mm). El éxito de una configuración de voltajes se calcula con la función objetivo ponderada normalizada J<sub>v2</sub> en el rango [0, 1] más penalizaciones. Esta función combina la transmisión (con una escala de hits parametrizada para evitar zonas ciegas), el acercamiento del haz al detector (distancia media del 10% de partículas más cercanas), la fracción de partículas que cruzan el plano, el offset radial del haz respecto al centro, el halo del haz, la kurtosis de la distribución, la divergencia angular y la emitancia. Durante el screening por RK4 se añade además una penalización lineal proporcional a la fracción de partículas colisionadas contra la pared."
    )
    story.append(Paragraph(p3_2_1, body_style))
    
    # Figure 1: fig_rk4_trayectorias_3d.png
    img_path_1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../outputs/report_figures/gemelo_db_v2/fig_rk4_trayectorias_3d.png'))
    add_image_or_placeholder(story, img_path_1, 420, 210,
                             "Figura 1: Trayectorias RK4 del gemelo central (bender7d, mejor configuración del estudio Search v2.1) en el plano de flexión XZ, con contornos de potencial en Y = 75 mm y la geometría del colimador extraída del archivo PA.", caption_style)
    
    story.append(Paragraph("3.3 Hiperparámetros", h2_style))
    
    # Keep the table as it was but updated with wording from md
    hp_headers = [Paragraph("<b>Parámetro</b>", meta_label_style),
                  Paragraph("<b>Valor</b>", meta_label_style),
                  Paragraph("<b>Justificación Física</b>", meta_label_style)]
    hp_data = [
        hp_headers,
        [Paragraph("Paso de integración (dt)", body_style),
         Paragraph("1.0 × 10<sup>-8</sup> s", body_style),
         Paragraph("Avance de ~0.6 mm/paso, permitiendo resolver la trayectoria en gaps estrechos de forma sub-milimétrica.", body_style)],
        [Paragraph("Screening en dos etapas", body_style),
         Paragraph("Etapa A (50 part. / 1500 pasos / semilla fija 42)<br/>Etapa B (200 part. / 3000 pasos / semilla 1234 / dt=5e-9)", body_style),
         Paragraph("Evita sobreajustar a una única muestra del haz y concentra el costo fino (Etapa B) en el top-30 de la Etapa A.", body_style)],
        [Paragraph("Margen de contacto", body_style),
         Paragraph("0.5 mm", body_style),
         Paragraph("Margen físicamente correcto sobre las paredes reales de la máscara del PA.", body_style)],
        [Paragraph("Transmisión (H0)", body_style),
         Paragraph("15 (con término lineal &lambda; = 2)", body_style),
         Paragraph("Evita saturar antes de los 30-60 hits, complementado con el término lineal que permite diferenciar soluciones en el régimen alto de hits.", body_style)],
        [Paragraph("GP seeds (n_gp_seeds)", body_style),
         Paragraph("5", body_style),
         Paragraph("Número de candidatos evaluados en el GP real en cada lazo para explotar covarianza antes del screening rápido.", body_style)]
    ]
    hp_table = Table(hp_data, colWidths=[120, 120, 264])
    hp_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, bg_light]),
        ('GRID', (0,0), (-1,-1), 0.5, border_color),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(hp_table)
    story.append(Spacer(1, 0.4 * cm))
    
    story.append(Paragraph("3.4 Validación del Modelo", h2_style))
    p3_4_1 = (
        "El criterio adoptado para aceptar cualquier cambio al modelo constó de dos condiciones: primero, que el cambio "
        "pudiera reproducir los resultados de la versión anterior (que no afectara lo que ya funcionaba); segundo, que "
        "representara una mejora medible frente al original, sin que el costo de implementación o ejecución superara el "
        "presupuesto disponible dentro de la hackathon."
    )
    story.append(Paragraph(p3_4_1, body_style))
    
    p3_4_2 = (
        "Este criterio se aplicó de forma sistemática mediante <code>validate_rk4_filter.py</code>, un arnés que reevalúa "
        "cualquier versión del filtro RK4 contra el historial completo de resultados reales de SIMION (288 configuraciones con "
        "verdad de terreno conocida, sin ejecutar ninguna corrida nueva) y reporta la correlación obtenida y cuántos "
        "<i>hitters</i> reales caen en su <i>top-30</i>. Con este método se determinó, por ejemplo, que terminar la trayectoria "
        "de una partícula al tocar la pared —un cambio que parecía más realista por replicar el comportamiento de SIMION— "
        "en realidad empeoraba el ranking (7–9 de 61 hitters reales en su top-30, frente a 16–17 sin esa terminación) [C3], "
        "por lo que se descartó pese a la intuición inicial. Con el mismo criterio se aceptó el cambio de interpolador de campo "
        "(resultados idénticos, 1.6–2.6x más rápido) y la reducción del chequeo de colisiones a cada 3 pasos en lugar de cada "
        "paso (mismo puntaje, ~4.2x menos costo); revisar cada 5 pasos, en cambio, perdía señal y fue rechazado."
    )
    story.append(Paragraph(p3_4_2, body_style))
    
    p3_4_3 = (
        "La misma disciplina se aplicó a los cambios de física: al reemplazar la geometría STL por la extraída del archivo PA, "
        "la aceptación no se basó en que la nueva geometría “pareciera más correcta”, sino en su desempeño frente a 5 casos canónicos "
        "con resultado de SIMION conocido (dos vivos, dos muertos, uno limítrofe) y frente al mismo conjunto de 288 configuraciones "
        "usado por <code>validate_rk4_filter.py</code>, exigiendo que igualara o superara a la versión anterior antes de promoverla a "
        "producción [C10]. De igual forma se evaluó el margen de contacto (1.5, 1.0, 0.5 mm) y la resolución de la malla de campo "
        "(1 mm nativa del PA frente a 2.5 mm): en el caso de la resolución, la malla más fina no mejoró el ranking pero sí multiplicó "
        "el costo de cómputo, por lo que se conservó la opción más económica [C10]."
    )
    story.append(Paragraph(p3_4_3, body_style))
    
    p3_4_4 = (
        "Fuera del filtro, el modelo completo se verificó contra SIMION en regiones fuera del experimento real: para voltajes "
        "aleatorios lejos de la cuenca de solución, el RK4 predice que el haz muere lejos del detector (~325 mm en promedio) y "
        "SIMION da un resultado equivalente (~324–338 mm) [C5]. Esta es una verificación económica pero informativa: si el gemelo "
        "y la realidad no coinciden en el caso trivial, no resulta razonable confiar en el caso difícil."
    )
    story.append(Paragraph(p3_4_4, body_style))
    
    story.append(Paragraph("3.5 Búsqueda Dinámica: Exploración Radial desde Cero", h2_style))
    p3_5_1 = (
        "Aun con el filtro RK4 y la reparametrización del bender, la evidencia de cuencas dejaba abierto un problema: el punto de "
        "arranque determinaba el resultado de la corrida. La cuenca asimétrica de alto rendimiento es un canal estrecho y empinado "
        "en el plano de voltajes del bender; un optimizador con radio de búsqueda constante y grande salta por encima de ese canal "
        "sin registrar hits, y queda atrapado en la cuenca simétrica, más ancha pero menos eficiente (~30 hits)."
    )
    story.append(Paragraph(p3_5_1, body_style))
    
    p3_5_2 = (
        "El diseño de la solución partió de una revisión crítica del historial: el análisis cruzado de las bases de datos previas "
        "reveló que los primeros resultados altos (95 y 84 hits en la campaña TES-GP-GP6D) se habían obtenido con los límites de "
        "búsqueda restringidos manualmente a la zona óptima —el electrodo V12 estaba acotado a [-1000, -604] V, bloqueando el resto "
        "del espacio—, es decir, no correspondían a una búsqueda real desde cero. Al eliminar ese sesgo y abrir las cotas al rango "
        "completo de hardware, el modelo 8D colapsó a 38 hits (Search v1): el volumen de 2000<sup>8</sup> combinaciones lo dominó. "
        "Ese resultado es el punto de partida legítimo frente al cual debe medirse todo lo que sigue."
    )
    story.append(Paragraph(p3_5_2, body_style))
    
    p3_5_3 = (
        "La solución implementada es un plan de búsqueda dinámica en 6 rondas que automatiza la transición de exploración global "
        "a explotación local, con cuatro componentes:<br/>"
        "1. <b>Inicialización insesgada:</b> Todos los voltajes optimizables arrancan en exactamente 0.0 V, forzando una exploración radial desde el centro de la línea de haz, sin heredar sesgo de corridas anteriores.<br/>"
        "2. <b>Templado (annealing) de perturbaciones:</b> La desviación estándar de las perturbaciones alrededor de la mejor configuración histórica se reduce geométricamente por ronda, de 500 V hasta un piso de refinamiento de 15 V.<br/>"
        "3. <b>Encogimiento dinámico de límites:</b> Las cotas locales del bender se estrechan alrededor del mejor punto histórico, del 100% del ancho en la ronda 1 al 40% en la ronda 6.<br/>"
        "4. <b>Desacople del espacio del GP:</b> Para evitar que Optuna degradara a muestreo aleatorio por el cambio de límites en <code>suggest_float</code>, el sampler consulta siempre límites estáticos en la base de datos, y el encogimiento se aplica exclusivamente en la generación offline de perturbaciones que alimenta el filtro RK4."
    )
    story.append(Paragraph(p3_5_3, body_style))
    
    p3_5_4 = (
        "El costo del screening también se administra por etapas mediante el parámetro de penalización cinemática &kappa;: en las "
        "rondas 1 y 2 el cálculo cinemático (aberración, emitancia, Twiss) se desactiva por completo (&kappa; = 0) y el puntaje se "
        "concentra en si los iones cruzan o no el plano, lo que redujo el tiempo de screening de ~120 s a menos de 10 s por ronda; "
        "en la ronda 3 se activa con peso moderado (&kappa; = 10) para orientar la búsqueda hacia el enfoque; y en las rondas 4 "
        "a 6 se endurece (&kappa; = 30) para pulir el spot."
    )
    story.append(Paragraph(p3_5_4, body_style))
    
    p3_5_5 = (
        "El resultado validó el diseño: arrancando en 0.0 V y explorando el rango completo de hardware, la versión 7D localizó y "
        "refinó de forma orgánica el canal de transmisión asimétrico —63 hits (trial #53) en 61 trials—, con voltajes seguros para "
        "el hardware (deflectores prácticamente apagados: V<sub>15</sub> = -48 V, V<sub>18</sub> = 0.2 V). El contraste con la "
        "solución 6D de fuerza bruta (113 hits, pero con tres electrodos saturados en sus límites de &plusmn;1000 V, con riesgo de "
        "arqueo de vacío) es parte de la razón por la que la selección del gemelo central no se decidió únicamente por el pico de hits."
    )
    story.append(Paragraph(p3_5_5, body_style))
    
    # Figure 2: Esquema de búsqueda dinámica en 6 rondas.png
    img_path_2 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../outputs/report_figures/Esquema de búsqueda dinámica en 6 rondas.png'))
    add_image_or_placeholder(story, img_path_2, 420, 180,
                             "Figura 2: Esquema de búsqueda dinámica en 6 rondas: templado geométrico de las perturbaciones (500 a 15 V), encogimiento de límites del bender (100% a 40%) y activación por etapas de la penalización cinemática &kappa;. Elaboración propia a partir de la implementación en optimizer.py.", caption_style)
    
    story.append(PageBreak())
    
    # ---------------------------------------------------------
    # SECTION 4: RESULTADOS FÍSICOS Y NUMÉRICOS
    # ---------------------------------------------------------
    story.append(Paragraph("4. Resultados Físicos y Numéricos del Acoplamiento", h1_style))
    
    p4_intro = (
        "En esta sección se detallan los resultados experimentales y numéricos obtenidos al "
        "conectar el gemelo digital RK4 con la simulación real de SIMION en el lazo cerrado "
        "de optimización bayesiana. Se evalúa el comportamiento del modelo tanto en su región "
        "informativa como en las zonas de extrapolación, además de analizar la incertidumbre "
        "del proceso gaussiano."
    )
    story.append(Paragraph(p4_intro, body_style))
    
    story.append(Paragraph("4.1 Correlación Predicho contra Real en la Región Informativa", h2_style))
    
    p4_corr_text = (
        "El gemelo digital se evaluó en la región de voltajes que presenta viabilidad física real "
        "(n = 153 candidatos con señal de transmisión). Los coeficientes de correlación calculados "
        "entre el costo predicho por el gemelo RK4 y el costo obtenido en SIMION real fueron un "
        "coeficiente de Pearson de <b>+0.287</b> y un coeficiente de Spearman de <b>+0.172</b>."
    )
    story.append(Paragraph(p4_corr_text, body_style))
    
    p4_corr_text_2 = (
        "<i>Interpretación física:</i> Aunque la correlación fina en el ordenamiento dentro de la cuenca es "
        "moderada (lo que refleja que las fluctuaciones menores del flujo no son recreadas idénticamente debido "
        "a las diferencias estocásticas de SIMION y la grilla de potenciales de 2.5 mm), el gemelo destaca como "
        "un <b>excelente clasificador grueso</b>. Separa con gran confiabilidad las configuraciones muertas "
        "de aquellas con transmisión real (hits > 0). Esto permite al loop concentrar todo su presupuesto de simulación "
        "real en candidatos promisorios, actuando como un filtro altamente efectivo."
    )
    story.append(Paragraph(p4_corr_text_2, body_style))
    
    # Validation image
    img_path_val = os.path.abspath(os.path.join(os.path.dirname(__file__), '../outputs/report_figures/gemelo_v2_bender7d/fig_validacion_gemelo.png'))
    if os.path.exists(img_path_val):
        story.append(Image(img_path_val, width=280, height=228))
        story.append(Paragraph("Figura 4.1: Correlación entre la predicción del Gemelo RK4 y el resultado real de SIMION en la región informativa. Los puntos de colores muestran los candidatos exitosos con hits reales de transmisión en SIMION.", caption_style))
    else:
        story.append(Paragraph("<i>[Figura 4.1: Correlación de validación del gemelo en región informativa - No encontrada]</i>", caption_style))
        
    story.append(Spacer(1, 0.3 * cm))
    
    story.append(Paragraph("4.2 Comportamiento Fuera de la Región de Entrenamiento", h2_style))
    
    p4_ext_text = (
        "Para verificar la robustez física y coherencia cualitativa del simulador intermedio RK4, se evaluó "
        "su respuesta en dos regiones extremas fuera del vecindario de soluciones conocidas:"
    )
    story.append(Paragraph(p4_ext_text, body_style))
    
    e1 = (
        "<b>1. Voltajes aleatorios alejados de la cuenca:</b> Para configuraciones de voltajes uniformes "
        "al azar lejos de la cuenca óptima, el gemelo RK4 predice de forma sistemática que el haz de iones "
        "choca tempranamente contra las paredes y muere a una distancia media de <b>~325 mm</b> (lejos del detector). "
        "SIMION real da exactamente el mismo resultado plano (~324-338 mm), validando el comportamiento asintótico."
    )
    story.append(Paragraph(e1, bullet_style))
    
    e2 = (
        "<b>2. Voltajes de control en cero:</b> Al apagar todos los voltajes optimizables de colimación y deflexión "
        "(fijando solo la fuente a +500 V y el detector a -2000 V), ambos simuladores colapsan el haz de iones, "
        "mostrando que las partículas impactan contra la pared en <b>z ~ 80 mm</b>. Esta concordancia geométrica "
        "directa fue la primera demostración del alineamiento físico tras corregir el desfase de unidades."
    )
    story.append(Paragraph(e2, bullet_style))
    
    story.append(PageBreak())
    
    story.append(Paragraph("4.3 Modelado de Incertidumbre y Extrapolación del Proceso Gaussiano", h2_style))
    
    p4_gp_text = (
        "El Proceso Gaussiano (GP) aproxima la función objetivo J_v2 en el espacio de voltajes de 8 dimensiones, "
        "modelando la incertidumbre predictiva a través de su desviación estándar (&sigma;). Se analizaron dos cortes "
        "unidimensionales correspondientes a las lentes de colimación horizontal (V3) y al doblador cuadrupolar (V9) "
        "para evaluar cómo se comporta el estimador de incertidumbre del GP:"
    )
    story.append(Paragraph(p4_gp_text, body_style))
    
    gp1 = (
        "<b>• Lentes de colimación (V3):</b> Presentan una cuenca de transmisión ancha y suave. La incertidumbre del "
        "GP (&plusmn;2&sigma;, representada por el área sombreada azul) es muy amplia en los bordes lejanos del espacio (&plusmn;1000 V) "
        "y se contrae gradualmente al aproximarse al óptimo (V3 ~ -270 V), debido a una longitud de escala de correlación amplia "
        "(l<sub>s</sub> = 380 V) que facilita la exploración suave."
    )
    story.append(Paragraph(gp1, bullet_style))
    
    gp2 = (
        "<b>• Bender Cuadrupolar (V9):</b> Presenta una sensibilidad física crítica. Al requerir una longitud de escala de "
        "correlación mucho más corta (l<sub>s</sub> = 100 V), la incertidumbre del GP se contrae de forma abrupta e hiper-local "
        "en las cercanías del óptimo (V9 ~ -455 V) y se expande violentamente al alejarse de este. Esto ilustra la alta "
        "selectividad de la cuenca del doblador y cómo el GP guía de forma inteligente la optimización Bayesiana."
    )
    story.append(Paragraph(gp2, bullet_style))
    
    story.append(Spacer(1, 0.4 * cm))
    
    # Side-by-side uncertainty images (V3 and V9)
    img_path_v3 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../outputs/report_figures/gemelo_v2_bender7d/fig_incertidumbre_V3.png'))
    img_path_v9 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../outputs/report_figures/gemelo_v2_bender7d/fig_incertidumbre_V9.png'))
    
    row_imgs = []
    row_caps = []
    
    if os.path.exists(img_path_v3):
        row_imgs.append(Image(img_path_v3, width=240, height=170))
        row_caps.append(Paragraph("Figura 4.2: Incertidumbre y corte del GP para la lente de colimación V3. Muestra una cuenca de correlación ancha y suave.", caption_style))
    else:
        row_imgs.append(Paragraph("<i>[Figura 4.2: Corte de incertidumbre GP en V3 - No encontrada]</i>", caption_style))
        row_caps.append(Paragraph("", caption_style))
        
    if os.path.exists(img_path_v9):
        row_imgs.append(Image(img_path_v9, width=240, height=170))
        row_caps.append(Paragraph("Figura 4.3: Incertidumbre y corte del GP para el electrodo del doblador V9. Muestra una cuenca hiper-local y muy selectiva.", caption_style))
    else:
        row_imgs.append(Paragraph("<i>[Figura 4.3: Corte de incertidumbre GP en V9 - No encontrada]</i>", caption_style))
        row_caps.append(Paragraph("", caption_style))
        
    gp_table_data = [row_imgs, row_caps]
    gp_table = Table(gp_table_data, colWidths=[252, 252])
    gp_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('PADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(gp_table)
    
    story.append(PageBreak())
    
    # ---------------------------------------------------------
    # SECTION 5: CONTROL DEL HAZ Y CONSIGNA INVERSA
    # ---------------------------------------------------------
    story.append(Paragraph("5. Control del Haz: Dirección y Consigna Inversa", h1_style))
    
    p5_intro = (
        "El control efectivo del haz de iones requiere la sintonía precisa de los voltajes de enfoque "
        "y deflexión. En esta sección se presentan los resultados obtenidos al aplicar las tareas "
        "formales de <b>Dirección (Tarea A)</b> y de <b>Consigna Inversa (Tarea B)</b>, evaluando las "
        "predicciones del gemelo digital contra los resultados reales de SIMION, e introduciendo los "
        "algoritmos de optimización global y local utilizados."
    )
    story.append(Paragraph(p5_intro, body_style))
    
    story.append(Paragraph("5.1 Tarea A: Dirección del Haz", h2_style))
    
    p5_task_a = (
        "El objetivo de la Tarea A consistió en enfocar y doblar el haz de iones Si+ de forma óptima desde la fuente "
        "hasta centrarlo en la ventana del detector. El lazo cerrado del gemelo digital identificó la siguiente configuración "
        "óptima de voltajes:"
    )
    story.append(Paragraph(p5_task_a, body_style))
    
    v_opt_text = (
        "<b>Voltajes Óptimos:</b> V<sub>3</sub> = 315.6 V, V<sub>6</sub> = -333.7 V, V<sub>9</sub> = -418.1 V, "
        "V<sub>10</sub> = 18.9 V, V<sub>11</sub> = -8.5 V, V<sub>12</sub> = -912.6 V, V<sub>15</sub> = 93.2 V, "
        "V<sub>18</sub> = -581.0 V."
    )
    story.append(Paragraph(v_opt_text, body_style))
    
    p5_task_a_res = (
        "Con estos valores, el gemelo digital RK4 predijo un alcance (<i>reach</i>) de <b>0.65</b> (el 65% de las "
        "partículas del haz cruzan la zona útil del detector) con un valor final de la función de costo J<sub>v2</sub> = 0.161. "
        "Al volar el haz en el simulador SIMION real, se midieron <b>63 &plusmn; 5 hits promedio</b> (de 500 partículas iniciales), "
        "lo que representa una tasa de transmisión real del <b>12.7% &plusmn; 1.0%</b> (aproximadamente el 60% de la capacidad de "
        "transmisión máxima del canal medida a 105 hits). La forma real del haz en el detector se ilustra en la Figura 5.3."
    )
    story.append(Paragraph(p5_task_a_res, body_style))
    
    story.append(Paragraph("5.2 Tarea B: Consigna Inversa del Haz", h2_style))
    
    p5_task_b = (
        "La Tarea B consistió en calcular los voltajes deflectores necesarios en las placas horizontal (V15) y "
        "vertical (V18) para posicionar el haz de iones en una consigna de desviación específica sobre el detector. "
        "Se aplicó un <b>método de optimización local de Gauss-Newton amortiguado con búsqueda de línea (Line Search)</b> "
        "sobre el gemelo digital RK4. Este algoritmo estima rápidamente el Jacobiano local 2x2 (dOffset/dV) para sintonizar "
        "los voltajes deflectores de manera casi instantánea."
    )
    story.append(Paragraph(p5_task_b, body_style))
    
    p5_task_b_res = (
        "El optimizador convergió en el gemelo a voltajes deflectores de <b>V<sub>15</sub> = 401 V</b> y <b>V<sub>18</sub> = -1000 V</b>, "
        "prediciendo una desviación en el plano del detector (z = 390 mm) de <b>(-1.90, +1.37) mm</b>. "
        "Al validar esta configuración en SIMION real, se obtuvo un desplazamiento medido de <b>(+10.6, +1.4) mm</b>, "
        "con una transmisión de 52 hits. Esto arrojó una brecha real-objetivo final de <b>10.69 mm</b>. La calibración "
        "general y curvas lineales del control de dirección horizontal y vertical se detallan en la Figura 5.1."
    )
    story.append(Paragraph(p5_task_b_res, body_style))
    
    story.append(Paragraph("5.3 Métodos de Optimización Utilizados", h2_style))
    
    p5_methods = (
        "El proyecto combinó dos estrategias de optimización complementarias:"
    )
    story.append(Paragraph(p5_methods, body_style))
    
    opt1 = (
        "<b>• Optimización Global (Lazo Cerrado Híbrido):</b> Para explorar el espacio de 8 dimensiones y "
        "escapar de las cuencas locales de polaridad opuesta (separadas por más de 1200 V, ver Sección 2), se combinó "
        "el sampler TPE (Tree-structured Parzen Estimator) y el GP (Gaussian Process). El lazo Bayesiano evalúa semillas del GP "
        "combinadas con perturbaciones locales aleatorias a través del integrador RK4, promoviendo selectivamente una combinación "
        "del top-10 del RK4 y las 5 semillas informadas del GP para evaluar en SIMION real."
    )
    story.append(Paragraph(opt1, bullet_style))
    
    opt2 = (
        "<b>• Optimización Local (Calibración y Consigna Inversa):</b> Se utilizó el algoritmo Gauss-Newton "
        "amortiguado sobre el gemelo digital RK4. Dado que el gemelo evalúa en milisegundos y calcula gradientes locales continuos, "
        "el optimizador local aproxima la consigna deseada en escasos milisegundos, superando la limitación de la dispersión de "
        "SIMION real."
    )
    story.append(Paragraph(opt2, bullet_style))
    
    story.append(PageBreak())
    
    # --- PAGE 10: CONTROL FIGURES ---
    story.append(Paragraph("Figuras y Calibración del Sistema de Control", h1_style))
    
    # 5.1 fig_control_consigna_inversa.png (horizontal / vertical calibration)
    img_path_ctrl = os.path.abspath(os.path.join(os.path.dirname(__file__), '../outputs/report_figures/gemelo_v2_bender7d/fig_control_consigna_inversa.png'))
    if os.path.exists(img_path_ctrl):
        story.append(Image(img_path_ctrl, width=440, height=202))
        story.append(Paragraph("Figura 5.1: Calibración del control de dirección y consigna inversa del haz. Paneles horizontal (V15 vs x_off) y vertical (V18 vs y_off) que derivan el ajuste de control inverso para guiar la desviación en el plano del detector.", caption_style))
    else:
        story.append(Paragraph("<i>[Figura 5.1: Calibración de deflectores y consigna inversa - No encontrada]</i>", caption_style))
        
    story.append(Spacer(1, 0.4 * cm))
    
    # Side-by-side: fig_convergencia_gemelo.png and fig_haz_en_detector.png
    img_path_conv = os.path.abspath(os.path.join(os.path.dirname(__file__), '../outputs/report_figures/fig_convergencia_gemelo.png'))
    img_path_haz = os.path.abspath(os.path.join(os.path.dirname(__file__), '../outputs/report_figures/gemelo_v2_bender7d/fig_haz_en_detector.png'))
    
    r_imgs = []
    r_caps = []
    
    if os.path.exists(img_path_conv):
        import struct
        h_conv = 172
        try:
            with open(img_path_conv, 'rb') as f:
                data = f.read(24)
                if data[:8] == b'\x89PNG\r\n\x1a\n' and data[12:16] == b'IHDR':
                    w_orig, h_orig = struct.unpack('>ii', data[16:24])
                    h_conv = int((h_orig / w_orig) * 230)
        except Exception:
            pass
        r_imgs.append(Image(img_path_conv, width=230, height=h_conv))
        r_caps.append(Paragraph("Figura 5.2: Convergencia de optimización global. Compara la evolución del costo J<sub>v2</sub> entre el optimizador TPE y el GP-seeded Bayesiano.", caption_style))
    else:
        r_imgs.append(Paragraph("<i>[Figura 5.2: Convergencia de optimización - No encontrada]</i>", caption_style))
        r_caps.append(Paragraph("", caption_style))
        
    if os.path.exists(img_path_haz):
        r_imgs.append(Image(img_path_haz, width=240, height=100))
        r_caps.append(Paragraph("Figura 5.3: Forma real del haz (SIMION) para el óptimo de la Tarea A. Muestra la dispersión de splats e impacto en la ventana verde del detector.", caption_style))
    else:
        story.append(Paragraph("<i>[Figura 5.3: Splats en detector de la mejor configuración - No encontrada]</i>", caption_style))
        r_caps.append(Paragraph("", caption_style))
        
    table_ctrl_data = [r_imgs, r_caps]
    table_ctrl = Table(table_ctrl_data, colWidths=[252, 252])
    table_ctrl.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('PADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(table_ctrl)
    
    story.append(PageBreak())
    
    # ---------------------------------------------------------
    # SECTION 6: METODOLOGÍA
    # ---------------------------------------------------------
    story.append(Paragraph("6. Metodología y Proceso de Desarrollo", h1_style))
    
    story.append(Paragraph("6.1 Diagnóstico Inicial y Primera Reformulación del Flujo", h2_style))
    p6_1 = (
        "La implementación inicial del gemelo digital siguió el flujo básico descrito en el documento de la hackathon, "
        "combinando Optuna y SIMION. Esta implementación no presentó dificultades técnicas; el obstáculo principal fue otro: "
        "la búsqueda de una solución en un espacio de parámetros prácticamente inexplorado."
    )
    story.append(Paragraph(p6_1, body_style))
    
    p6_2 = (
        "Una limitación identificada en el flujo original es que este invoca a SIMION una única vez por candidato, "
        "sobre el punto que optimiza la transmisión. Sin embargo, el Proceso Gaussiano (GP) subyacente está diseñado "
        "para mejorar el conocimiento de la distribución de la transmisión en función de los candidatos, no para "
        "optimizar la forma del haz —que este se doble 90 grados y llegue relativamente colimado—. La magnitud de esta "
        "limitación quedó cuantificada al final del proyecto: la banda de voltajes que produce hits ocupa apenas "
        "~1 × 10<sup>-9</sup> del volumen del espacio de búsqueda [C2], y las primeras 26 o más evaluaciones de "
        "SIMION con voltajes aleatorios no produjeron ningún hit [C8]."
    )
    story.append(Paragraph(p6_2, body_style))
    
    p6_3 = (
        "A esto se suma que SIMION no está optimizado para ejecutar múltiples simulaciones de voltajes: cada evaluación "
        "cuesta ~5.77 s, dominados no por la física sino por el arranque del proceso y la carga del arreglo de potenciales "
        "de 290 MB [C1]. Esto implica que una búsqueda exhaustiva sobre SIMION resulta prohibitivamente costosa."
    )
    story.append(Paragraph(p6_3, body_style))
    
    p6_4 = (
        "La primera modificación al flujo buscó explotar la física del experimento para explorar el espacio de "
        "soluciones de forma más eficiente, mediante un esquema en el que Optuna propone candidatos que son evaluados "
        "por un analizador de trayectorias basado en RK4, encargado de rankear la viabilidad del haz antes de invocar "
        "a SIMION. En su versión inicial, este analizador solo evaluaba si el punto final de la trayectoria del ion caía "
        "dentro del detector. El desarrollo posterior mostró que el analizador debía funcionar como una simulación de "
        "trayectoria equivalente a SIMION, no para explorar el espacio real, sino para estimar el potencial de cada "
        "candidato de producir un haz físicamente realista."
    )
    story.append(Paragraph(p6_4, body_style))
    
    p6_5 = (
        "El simulador RK4 se desarrolló de forma incremental. Las primeras etapas consistieron en construir las clases "
        "necesarias para almacenar y procesar los cálculos del RK4, seguidas de la vectorización completa de las "
        "funciones mediante <code>numpy</code> para permitir su ejecución en paralelo. La función de calificación del haz "
        "pasó por varias iteraciones: un puntaje inicial unidimensional a lo largo del eje de viaje (sin distinguir "
        "la desviación del haz en los otros dos ejes); luego un puntaje contra el volumen real del detector en 3D, junto "
        "con detección de colisiones contra la geometría de los electrodos (basada en un índice de distancias construido "
        "desde archivos STL) y penalización de partículas perdidas; posteriormente, la corrección de la escala de la "
        "recompensa (de 30 a 150 mm), tras medir que a las distancias reales de los candidatos el término de recompensa "
        "aportaba señal prácticamente nula; y finalmente, la alineación del puntaje del RK4 con el mismo objetivo "
        "medido en SIMION —la distancia al detector del 10% más cercano del haz, más un término de transmisión y una "
        "penalización por proximidad a paredes—."
    )
    story.append(Paragraph(p6_5, body_style))
    
    story.append(Paragraph("6.2 Selección del Modelo de Simulación", h2_style))
    p6_6 = (
        "El analizador tiene como función explorar el espacio de búsqueda con mayor profundidad, con el fin de "
        "identificar tanto candidatos claramente inviables como candidatos que producen un haz con las características "
        "deseadas; esta función es distinta de la del GP, que optimiza la transmisión y no la forma del haz."
    )
    story.append(Paragraph(p6_6, body_style))
    
    p6_7 = (
        "Esto plantea la pregunta de qué tipo de simulador permite investigar el espacio de forma útil sin incurrir en "
        "un costo superior al de SIMION. Se evaluaron dos alternativas: modelos estocásticos de aprendizaje (redes "
        "neuronales o modelos tipo LLM) y aproximaciones numéricas como RK4. Se optó por un simulador RK4 por dos "
        "razones principales:<br/>"
        "1. <b>Eficiencia en Paralelo:</b> El método numérico puede optimizarse mediante aritmética matricial "
        "vectorizada con <code>numpy</code>, permitiendo calcular múltiples trayectorias en paralelo sin recurrir a bucles "
        "costosos en Python: en su versión final, el simulador evalúa ~239 candidatos en el mismo tiempo de pared en que "
        "SIMION evalúa ~10 (~0.25 s por candidato frente a ~5.77 s, una relación de ~23x) [C1]. Además, el método numérico "
        "aprovecha una propiedad física conocida: como la ecuación del potencial es lineal, el campo de cualquier "
        "combinación de voltajes es la combinación lineal de 19 campos base extraídos de SIMION una sola vez, por lo que el "
        "campo del gemelo es esencialmente exacto por construcción.<br/>"
        "2. <b>Costo de Datos Inviable para Modelos Estadísticos:</b> Un modelo estocástico habría requerido datos "
        "suficientes para construir conjuntos de entrenamiento y prueba. Se estimó ese costo [C2]: dado que un regresor "
        "necesita ejemplos positivos (candidatos con transmisión) para aprender algo más que “todo da cero”, y considerando "
        "la banda de voltajes donde aparecen los hits, un muestreo aleatorio produciría en promedio un positivo cada "
        "~8.6 × 10<sup>8</sup> evaluaciones de SIMION. Incluso bajo el supuesto más favorable sobre el tamaño de la cuenca "
        "de solución, construir un conjunto de entrenamiento con apenas ~10 positivos habría costado ~41,000 evaluaciones "
        "—unas 85 veces el presupuesto total de SIMION consumido por el proyecto completo (484 evaluaciones, ~46 minutos) [C7]—. "
        "El modelo estocástico resultaba, en la práctica, inalcanzable dentro del plazo de la hackathon."
    )
    story.append(Paragraph(p6_7, body_style))
    
    story.append(Paragraph("6.4 Incorporación de Conocimiento Físico del Sistema", h2_style))
    p6_8 = (
        "Aplicando la misma lógica de explotar el conocimiento físico del experimento, se refinó la comprensión del "
        "cuadrupolo como un sistema correlacionado, en lugar de cuatro electrodos independientes. Esto condujo al "
        "análisis de campo del bender (<i>bender field analysis</i>) para obtener una mejor propuesta inicial: se ubicaron "
        "los cuatro electrodos del doblador a partir de sus campos base, se midió la deflexión que aporta cada uno por "
        "voltio a lo largo de la trayectoria del haz, y se derivó de ahí el patrón correlacionado (pares diagonales de "
        "polaridad opuesta) y su intensidad."
    )
    story.append(Paragraph(p6_8, body_style))
    
    p6_9 = (
        "El resultado respalda esta decisión: todos los candidatos con hits reales encontrados en el proyecto descienden "
        "de perturbaciones de ese punto inicial físico, y cada estudio iniciado desde cero encontró su primer hit dentro "
        "de las primeras 3 a 7 evaluaciones de SIMION [C8], en un espacio donde la búsqueda aleatoria había dado "
        "sistemáticamente cero hits. Este enfoque fue posteriormente reemplazado en el desarrollo, pero constituyó un "
        "paso importante para conocer el espacio de soluciones y comenzar a entrenar gemelos con hits en lugar de "
        "buscar a ciegas."
    )
    story.append(Paragraph(p6_9, body_style))
    
    # Figure 3: outputs/bender_pattern_scan.png
    img_path_3 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../outputs/bender_pattern_scan.png'))
    add_image_or_placeholder(story, img_path_3, 400, 240,
                             "Figura 3: Barrido 1D sobre la dirección del patrón físico del cuadrupolo (V9..V12 = s por patrón, RK4 por lotes): la distancia media al detector colapsa de ~325 mm a su mínimo cerca de s = 450 V. Este punto físico es el origen común de todos los candidatos con hits reales del proyecto [C8][C12].", caption_style)
    
    story.append(Paragraph("6.5 Criterio de Aceptación de Cambios Metodológicos", h2_style))
    p6_10 = (
        "Estas modificaciones se realizaron aplicando un método sistemático de verificación entre versiones del modelo. "
        "Ante cada cambio propuesto se realizaban dos comprobaciones: primero, que la modificación pudiera reproducir los "
        "resultados de versiones anteriores ya validadas; segundo, que representara una mejora medible frente al "
        "original, sin que el costo o el tiempo de implementación comprometiera la viabilidad del proyecto dentro del "
        "plazo de la hackathon."
    )
    story.append(Paragraph(p6_10, body_style))
    
    p6_11 = (
        "Ejemplos concretos de este proceso, reproducibles con el código entregado:<br/>"
        "• <b>(i) Cambio de interpolador de campo:</b> Se aceptó porque produjo resultados idénticos 1.6–2.6 veces más rápido.<br/>"
        "• <b>(ii) Aceleración del chequeo de colisiones:</b> Revisar cada 3 pasos en lugar de cada paso se aceptó porque reprodujo exactamente los puntajes de la versión exhaustiva con ~4.2x menos costo, mientras que revisar cada 5 pasos ya perdía señal y fue rechazado.<br/>"
        "• <b>(iii) Terminación de partículas al tocar pared:</b> Un cambio que parecía correcto por replicar el comportamiento de SIMION se implementó y se descartó, ya que el arnés de validación la evaluó contra 288 configuraciones con resultado de SIMION conocido y mostró que empeoraba el ranking (capturaba 7–9 de 61 hitters reales en su top-30, frente a 16–17 sin terminación); en su lugar se mantuvo una penalización calibrada, que dejó el filtro en su mejor versión medida (Spearman +0.458, 2.7 veces más hitters en su top-30 que el azar [C3]). <code>validate_rk4_filter.py</code> reevalúa cualquier versión del filtro contra todo el historial de resultados reales sin gastar una sola corrida de SIMION, y <code>report_figures.py</code> regenera las figuras del informe a partir de las bases de datos persistidas."
    )
    story.append(Paragraph(p6_11, body_style))
    
    story.append(Paragraph("6.6 De la Geometría STL a la Geometría Extraída del Archivo PA", h2_style))
    p6_12 = (
        "La geometría de colisión usada para detectar choques contra electrodos partió inicialmente de archivos STL, "
        "ajustados al marco de referencia de SIMION mediante una transformación rígida (rotación y traslación) calculada "
        "por mínimos cuadrados sobre 6 puntos de referencia leídos manualmente del cursor de la interfaz de SIMION. "
        "Esta aproximación dejó un residuo medido de ~25 mm en los centroides de los 4 electrodos del cuadrupolo [C10], "
        "suficiente para invertir el juicio del filtro RK4 en esa zona: predecía transmisión ~0% para candidatos que "
        "en SIMION real transmitían ~78% del haz."
    )
    story.append(Paragraph(p6_12, body_style))
    
    p6_13 = (
        "La transformación y alineación rigurosa resolvió el problema (ver sección 6.6). La máscara metálica sólida "
        "se extrajo directamente del archivo PA0 (fijando en 1.0 los electrodos de interés y en 0.0 los excluidos). "
        "Validada contra los 5 casos canónicos, la geometría nueva predijo 388 de 395 partículas limpias para la mejor solución "
        "conocida, frente a 368 de 384 reales en SIMION; la anterior predecía cero y 85% de colisión fantasma. El margen "
        "se redujo de 1.5 mm a 0.5 mm."
    )
    story.append(Paragraph(p6_13, body_style))
    
    story.append(Paragraph("6.7 TPE como Motor Principal e Integración de Semillas del GP", h2_style))
    p6_14 = (
        "La mayor parte del desarrollo del proyecto se realizó con TPE (<i>Tree-structured Parzen Estimator</i>, el sampler "
        "por defecto de Optuna) combinado con el filtrado RK4; esta combinación, robusta y económica, produjo las "
        "primeras soluciones reales del gemelo. El GP (proceso gaussiano) ofrece una ventaja teórica que TPE no tiene "
        "—modela correlaciones entre variables mediante su kernel de covarianza, mientras que TPE estima cada parámetro de "
        "forma independiente—, pero consultar su modelo real (<code>ask() / sample_relative()</code>) cuesta ~100–300 ms "
        "y este costo crece con el historial, lo que resulta demasiado alto para generar los cientos de candidatos que "
        "RK4 filtra por iteración."
    )
    story.append(Paragraph(p6_14, body_style))
    
    p6_15 = (
        "La integración adoptada paga el costo real del GP solo unas pocas veces por iteración (n<sub>gp_seeds</sub>, típicamente 5): "
        "estas semillas informadas se mezclan dentro del mismo lote de candidatos que evalúa RK4, junto con perturbaciones "
        "económicas alrededor de ellas, de modo que se explora el espacio amplio con el mecanismo económico (TPE/RK4) "
        "mientras se explota el mejor punto identificado por el GP. El gráfico de convergencia muestra el resultado: "
        "durante las primeras ~70–80 evaluaciones ambos motores avanzan de forma comparable, incluso con TPE adelante "
        "durante buena parte del tramo; recién cerca de la evaluación #80 el estudio <i>GP-seeded</i> encuentra un mínimo "
        "considerablemente mejor y lo sostiene, mientras TPE se estanca en un valor peor."
    )
    story.append(Paragraph(p6_15, body_style))
    
    # Figure 4: outputs/report_figures/convergencia de optimizacion.png
    img_path_4 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../outputs/report_figures/convergencia de optimizacion.png'))
    add_image_or_placeholder(story, img_path_4, 400, 240,
                             "Figura 4: Convergencia comparada del costo J<sub>v2</sub> durante la campaña: TPE (gemelo_v2), GP-seeded (gemelo_db_v2) y bender6d. El GP-seeded se despega recién cerca de la evaluación #80: su ventaja depende del presupuesto adicional de evaluaciones, no de una superioridad por evaluación individual.", caption_style)
    
    story.append(Paragraph("6.8 Múltiples Óptimos de Polaridad Opuesta", h2_style))
    p6_16 = (
        "El espacio de soluciones que producen deflexión útil del haz es fino y sensible (~1 × 10<sup>-9</sup> del volumen "
        "total, [C2]), difícil de alcanzar por muestreo aleatorio. Al comparar los mejores candidatos de dos corridas "
        "<i>GP-seeded</i> independientes se identificaron dos regiones de solución con transmisión real, pero con la "
        "polaridad de varios electrodos invertida entre sí: una distancia de 1241 V (norma L2 sobre los 8 electrodos "
        "optimizables) entre ambos mejores candidatos, con signo opuesto en V<sub>3</sub>, V<sub>6</sub> y V<sub>9</sub> [C11]. "
        "La corrida que quedó en la cuenca de menor transmisión no escapó de ella durante el total de sus evaluaciones, "
        "pese a usar el mismo mecanismo de búsqueda que la otra corrida: la diferencia fue el punto de arranque, no el algoritmo."
    )
    story.append(Paragraph(p6_16, body_style))
    
    p6_17 = (
        "Esto se conecta directamente con el hecho de que ni TPE ni el GP, tal como están planteados, conocen la "
        "estructura física del problema (el patrón de pares diagonales correlacionados del cuadrupolo), por lo que "
        "ninguno cuenta con un mecanismo para evaluar, o saltar hacia, una cuenca mejor fuera de su vecindario de arranque."
    )
    story.append(Paragraph(p6_17, body_style))
    
    story.append(Paragraph("6.9 Reparametrización del Bender", h2_style))
    p6_18 = (
        "La limitación anterior dejaba una conclusión clara: el optimizador seguía explorando cuatro voltajes "
        "independientes cuando los datos mostraban que la solución se ubica sobre un patrón correlacionado. Para "
        "resolverla, se consolidaron los cuatro electrodos del doblador en tres variables de control físico: "
        "<b>A</b> como offset común, <b>B</b> como intensidad de flexión sobre el patrón diagonal ya derivado en el "
        "análisis de campo del bender, y <b>C</b> como asimetría dipolar horizontal, con el mapeo:<br/><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>V<sub>9</sub> = A + B + C</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>V<sub>10</sub> = A - B</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>V<sub>11</sub> = A - B</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>V<sub>12</sub> = A + B - C</b> [C12]<br/><br/>"
        "Con esto, la búsqueda dejó de operar sobre un espacio de 8 voltajes independientes y pasó a moverse sobre las "
        "coordenadas en las que el hardware realmente actúa."
    )
    story.append(Paragraph(p6_18, body_style))
    
    p6_19 = (
        "Las cotas de estas variables se manejaron con el mismo criterio: dejar que los datos determinen el ajuste. "
        "Cuando el mejor valor de B se acercaba al borde sin mostrar una meseta, se ampliaron los rangos de A y B "
        "(de &plusmn;500 a &plusmn;600 / &plusmn;800 V); la misma señal aparece ahora en C —4 de los 5 mejores hitters de "
        "una corrida quedaron en su cota de &plusmn;600 V [C12]—, por lo que la ampliación correspondiente queda "
        "pendiente como siguiente paso."
    )
    story.append(Paragraph(p6_19, body_style))
    
    p6_20 = (
        "El resultado justificó el cambio: el récord absoluto del proyecto surgió de esta parametrización (113 de "
        "500 iones con el bender6d; en una comparación directa a igual presupuesto, la versión reducida duplicó la "
        "media de hits del 8D original: 24.3 frente a 10.0 [C12]). De igual relevancia, la versión 7D encontró la "
        "cuenca asimétrica arrancando desde 0.0 V, sin ninguna semilla sesgada: el tipo de descubrimiento orgánico "
        "que anteriormente se consideraba poco probable sin dotar a la búsqueda de la estructura física del problema."
    )
    story.append(Paragraph(p6_20, body_style))
    
    story.append(Paragraph("6.10 Segunda Revisión de la Función Objetivo: Cambio de Régimen", h2_style))
    p6_21 = (
        "Corregir el objetivo una vez no garantizó que no debiera corregirse de nuevo. El término de transmisión "
        "saturante que resolvió el problema del régimen de 0 hits —una solución correcta en ese momento— se convirtió "
        "en un problema cuando la búsqueda maduró: en el régimen actual de 70–110 hits, un hit marginal aporta apenas "
        "~0.0005 al valor de J, y ninguna recalibración de esa forma funcional corrige la limitación [C13]."
    )
    story.append(Paragraph(p6_21, body_style))
    
    p6_22 = (
        "La consecuencia se observó directamente en los datos: el objetivo rankeaba una cuenca de 84 hits por encima "
        "de una de 105, porque los términos cosméticos del haz (offset, Twiss, colimación, halo) pesaban más que una "
        "diferencia de 21 hits reales. El optimizador no estaba fallando: estaba optimizando correctamente un criterio incorrecto."
    )
    story.append(Paragraph(p6_22, body_style))
    
    p6_23 = (
        "La corrección mantiene el término saturante —que sigue aportando gradiente en el arranque de la búsqueda— "
        "y le suma un término lineal de transmisión que no satura, re-normalizando el total para conservar la escala [0,1] "
        "del objetivo. Verificado contra los trials reales del proyecto, el nuevo objetivo reordena correctamente "
        "105 > 95 > 84 > 74 hits, con los términos cosméticos actuando como desempate entre configuraciones equivalentes [C13]. "
        "La revisión también permitió identificar dos problemas colaterales: el puntaje de penalización por vuelo fallido "
        "había quedado por debajo del nuevo peor caso válido, y el test unitario del haz perfecto estaba desactualizado. "
        "Ambos se corrigieron. La conclusión es que la función objetivo debe revisarse ante cada cambio de régimen de búsqueda."
    )
    story.append(Paragraph(p6_23, body_style))
    
    # Figure 5: Reordenamiento de cinco configuraciones.png
    img_path_5 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../outputs/report_figures/Reordenamiento de cinco configuraciones.png'))
    add_image_or_placeholder(story, img_path_5, 420, 150,
                             "Figura 5: Reordenamiento de cinco configuraciones reales del proyecto al pasar del objetivo v2.3 (saturante) al v2.4 (lineal, &lambda;=2). Con v2.3, el candidato de 84 hits rankeaba primero y el de 105 hits, último; con v2.4 el ranking sigue a los hits [C13].", caption_style)
    
    story.append(Paragraph("6.11 Evaluación Formal y Selección del Gemelo Central", h2_style))
    p6_24 = (
        "Para resolver la pregunta de cuál gemelo es mejor de forma objetiva, se construyó una rúbrica ejecutable que "
        "recorre todos los estudios persistidos y califica cada gemelo en siete dimensiones —admisibilidad física, "
        "dirección (hits del candidato recomendado), exactitud de ranking en la región informativa, linealidad de la "
        "consigna inversa, honestidad de la incertidumbre, alineación de gradiente contra SIMION y eficiencia de datos—, "
        "combinadas en un puntaje único [C14]."
    )
    story.append(Paragraph(p6_24, body_style))
    
    p6_25 = (
        "El resultado del ranking resume el estado del proyecto: el primer puesto (bender6d, 113 hits, puntaje 0.712) "
        "obtiene el mejor rendimiento crudo, pero presenta la peor exactitud de ranking fino de la tabla; el gemelo mejor "
        "calibrado (puntaje 0.631) encontró un óptimo menos extremo. Ningún gemelo domina simultáneamente en rendimiento "
        "pico y calibración fina."
    )
    story.append(Paragraph(p6_25, body_style))
    
    p6_26 = (
        "Como gemelo central del proyecto se seleccionó <b>bender7d</b>, del estudio Search v2.1 (puntaje 0.573, tercer "
        "lugar del ranking; mejor J de la parametrización 7D, con 74 hits en su configuración recomendada y 83 en su mejor "
        "hitter) [C14]. La decisión pondera que esta es la parametrización física completa (incluye la asimetría dipolar C), "
        "con una consigna inversa fuerte (I = 0.852) y la mayor honestidad de incertidumbre del conjunto —reconociendo "
        "que no posee el récord de hits, y que su eficiencia de datos es su punto más débil—. Todas las figuras del gemelo "
        "central se regeneraron a partir de su base de datos congelada en studies/."
    )
    story.append(Paragraph(p6_26, body_style))
    
    # Figure 6: Rúbrica de evaluación formal.png
    img_path_6 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../outputs/report_figures/Rúbrica de evaluación formal.png'))
    add_image_or_placeholder(story, img_path_6, 420, 160,
                             "Figura 6: Rúbrica de evaluación formal aplicada a todos los estudios persistidos (top 5, contribuciones ponderadas por término). El gemelo central seleccionado (bender7d, Search v2.1) obtiene 0.573 [C14].", caption_style)
    
    # Figure 7: Configuración de voltajes del gemelo central.png
    img_path_7 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../outputs/report_figures/Configuración de voltajes del gemelo central.png'))
    add_image_or_placeholder(story, img_path_7, 420, 160,
                             "Figura 7: Configuración de voltajes del gemelo central (mejor J del estudio Search v2.1, trial #83, 74 hits): V3 = +310, V6 = -372, bender A = -282 / B = -284 / C = +300 (V9..V12 expandidos), V15 = -67, V18 = -81.", caption_style)
    
    story.append(PageBreak())
    
    # ---------------------------------------------------------
    # SECTION 7: REFLEXIÓN
    # ---------------------------------------------------------
    story.append(Paragraph("7. Reflexión sobre la Confiabilidad del Gemelo Digital", h1_style))
    
    story.append(Paragraph("7.1 Confianza en el Gemelo Digital", h2_style))
    p7_1_1 = (
        "El desarrollo e implementación del gemelo digital RK4 ha permitido contrastar la teoría física "
        "con la práctica numérica. En esta sección se analizan con honestidad técnica los límites del "
        "modelo: hasta dónde es posible confiar en sus predicciones y dónde se identifican sus debilidades, "
        "cerrando con propuestas claras para futuros trabajos."
    )
    story.append(Paragraph(p7_1_1, body_style))
    
    p7_1_2 = (
        "El gemelo digital ha demostrado ser una herramienta robusta y confiable para las siguientes tareas:<br/>"
        "• <b>Identificación de cuencas útiles y vecindarios buenos:</b> El gemelo es excelente para encontrar "
        "mínimos locales y reducir drásticamente el espacio de búsqueda. Mientras la optimización aleatoria inicial "
        "en SIMION dio 0 hits en 26 corridas ciegas, el gemelo guiado físicamente encontró su primer hit en las primeras "
        "3 a 7 evaluaciones de cada estudio iniciado desde cero, logrando acumular 86 trials exitosos en total.<br/>"
        "• <b>Cribado masivo y descarte de malos candidatos:</b> El filtro RK4 clasifica con gran precisión "
        "los candidatos sin viabilidad. En la validación offline de 288 configuraciones, colocó 17 de los 61 hitters "
        "reales en su top-30 (un enriquecimiento de 2.7x frente al azar). Operativamente, todos los hits del proyecto "
        "provienen de candidatos promovidos por el RK4 en su top-10, lo que valida su rol como portero de la simulación real.<br/>"
        "• <b>Eficiencia y velocidad de cómputo:</b> El motor RK4 evalúa 239 candidatos en el mismo tiempo que SIMION "
        "corre 10 (0.25 s frente a 5.77 s por config, una aceleración de 23x), cumpliendo con creces los objetivos de throughput.<br/>"
        "• <b>Límites físicos y asintóticos cualitativos:</b> Ambos simuladores coinciden perfectamente en comportamientos "
        "extremos. Lejos de la cuenca útil, ambos muestran que el haz colisiona a lo largo del bender y muere a ~325 mm sin llegar al detector."
    )
    story.append(Paragraph(p7_1_2, body_style))
    
    # Figure 8: outputs/report_figures/gemelo_v2_bender7d/fig_haz_en_detector.png
    img_path_8 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../outputs/report_figures/gemelo_v2_bender7d/fig_haz_en_detector.png'))
    add_image_or_placeholder(story, img_path_8, 420, 180,
                             "Figura 8: Forma real del haz en SIMION para la mejor configuración del gemelo central: distribución de impactos (splats) en z y en el plano XY, en la ventana del detector.", caption_style)
    
    story.append(Paragraph("7.2 Limitaciones del Gemelo Digital", h2_style))
    p7_2_1 = (
        "A pesar de sus fortalezas, los datos muestran que el gemelo digital no es fiable para las siguientes tareas:<br/>"
        "• <b>Ordenamiento o ranking fino dentro de la cuenca:</b> Una vez dentro de la región con señal de transmisión, "
        "la correlación fina del gemelo con SIMION es débil (Spearman +0.172, Pearson +0.287). El gemelo no es capaz de discriminar "
        "cuál de dos candidatos 'buenos' es ligeramente mejor. Esto se debe a simplificaciones geométricas (malla de 2.5 mm) "
        "y a que el haz de SIMION se genera de forma estocástica en cada corrida.<br/>"
        "• <b>Estimación de la transmisión absoluta:</b> El gemelo tiende a sobreestimar de forma sistemática la transmisión "
        "absoluta de partículas (p. ej., prediciendo ~50% de supervivencia para voltajes que en SIMION real arrojan 1-2% de transmisión). "
        "El gemelo no sabe contar hits con precisión absoluta; debe usarse para ordenar vecindarios y filtrar.<br/>"
        "• <b>Representación del ruido físico:</b> El gemelo es determinista por construcción. Sin embargo, en SIMION "
        "el haz se genera aleatoriamente en cada corrida, provocando que un mismo juego de voltajes varíe entre 2, 4, 5, 8 "
        "y 10 hits en corridas idénticas (ruido de disparo). El gemelo no modela esta fluctuación intrínseca de la verdad de terreno."
    )
    story.append(Paragraph(p7_2_1, body_style))
    
    # Figure 9: outputs/report_figures/gemelo_v2_bender7d/fig_validacion_gemelo.png
    img_path_9 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../outputs/report_figures/gemelo_v2_bender7d/fig_validacion_gemelo.png'))
    add_image_or_placeholder(story, img_path_9, 420, 245,
                             "Figura 9: Validación del gemelo central en la región informativa: predicción de costo del RK4 frente al costo real medido en SIMION, coloreado por número de hits. El gemelo separa el vecindario bueno del malo, pero no ordena con precisión fina dentro del vecindario bueno.", caption_style)
    
    # Callout Box summarizing Section 7 and Recommendations
    callout_data_7 = [[
        Paragraph(
            "<b>[Siguientes Pasos Recomendados]</b><br/>"
            "1. <b>Reparametrización del GP:</b> Codificar el patrón de pares diagonales del cuadrupolo para que el GP "
            "reconozca la correlación física y escape de cuencas de polaridad opuesta.<br/>"
            "2. <b>Optimización por Etapas:</b> Sintonizar secuencialmente elemento por elemento en el orden en que "
            "el haz los recorre, convirtiendo el problema de 8D en dos subproblemas de 2D a 4D.<br/>"
            "3. <b>Optimización Local CMA-ES:</b> Reemplazar TPE por CMA-ES para el refinamiento local fino una vez "
            "identificada la cuenca con transmisión en el gemelo.",
            callout_text_style
        )
    ]]
    callout_table_7 = Table(callout_data_7, colWidths=[504])
    callout_table_7.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), bg_light),
        ('BOX', (0,0), (-1,-1), 0.5, border_color),
        ('LINELEFT', (0,0), (0,-1), 3.0, primary_color),
        ('PADDING', (0,0), (-1,-1), 10),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(callout_table_7)
    
    doc.build(story, canvasmaker=NumberedCanvas)
    print("Report PDF generated successfully!")

if __name__ == '__main__':
    create_report()

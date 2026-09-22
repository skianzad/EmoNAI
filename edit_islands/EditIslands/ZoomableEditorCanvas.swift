import SwiftUI
import UIKit

/// Photo + UIKit interaction overlay.
/// Cursor / marquee / eraser graphics update on the CALayer thread path (instant),
/// while mask edits only commit when the finger lifts.
struct ZoomableEditorCanvas: View {
    let composite: UIImage
    let overlay: UIImage?
    @Binding var scale: CGFloat
    @Binding var offset: CGSize
    let tool: EditorTool
    let eraserRadius: Int
    let onTap: (Int, Int) -> Void
    let onMarqueeSelect: (CGRect) -> Void
    let onEraseStroke: ([(x: Int, y: Int)]) -> Void

    @GestureState private var magnify: CGFloat = 1

    private let minScale: CGFloat = 1
    private let maxScale: CGFloat = 6

    var body: some View {
        GeometryReader { geo in
            let viewSize = geo.size
            if viewSize.width < 2 || viewSize.height < 2 {
                Color.black
            } else {
                let fit = fittedSize(image: composite.size, in: viewSize)
                let liveScale = min(max(scale * magnify, minScale), maxScale)

                ZStack {
                    Color.black

                    ZStack {
                        Image(uiImage: composite)
                            .resizable()
                            .interpolation(.high)
                            .frame(width: fit.width, height: fit.height)
                        if let overlay, overlay.size.width > 1, overlay.size.height > 1 {
                            Image(uiImage: overlay)
                                .resizable()
                                .interpolation(.none)
                                .frame(width: fit.width, height: fit.height)
                        }
                    }
                    .frame(width: fit.width, height: fit.height)
                    .scaleEffect(liveScale)
                    .offset(offset)
                    .position(x: viewSize.width / 2, y: viewSize.height / 2)
                    .allowsHitTesting(false)

                    // Instant UIKit overlays + touch handling (no SwiftUI body thrash)
                    InteractionOverlayRepresentable(
                        tool: tool,
                        eraserRadius: eraserRadius,
                        scale: liveScale,
                        offset: offset,
                        fit: fit,
                        imageSize: composite.size,
                        viewSize: viewSize,
                        onPan: { delta in
                            offset = CGSize(
                                width: offset.width + delta.width,
                                height: offset.height + delta.height
                            )
                        },
                        onTap: onTap,
                        onMarqueeSelect: onMarqueeSelect,
                        onEraseStroke: onEraseStroke
                    )
                    .frame(width: viewSize.width, height: viewSize.height)
                    .contentShape(Rectangle())
                    .simultaneousGesture(
                        MagnificationGesture()
                            .updating($magnify) { value, state, _ in state = value }
                            .onEnded { value in
                                let next = min(max(scale * value, minScale), maxScale)
                                scale = next <= 1.01 ? 1 : next
                                if scale <= 1.01 { offset = .zero }
                            }
                    )
                }
                .frame(width: viewSize.width, height: viewSize.height)
                .clipped()
            }
        }
    }

    private func fittedSize(image: CGSize, in bounds: CGSize) -> CGSize {
        let iw = max(image.width, 1)
        let ih = max(image.height, 1)
        let s = min(bounds.width / iw, bounds.height / ih)
        return CGSize(
            width: max(1, (iw * s).rounded(.down)),
            height: max(1, (ih * s).rounded(.down))
        )
    }
}

// MARK: - UIKit interaction (instant cursor + blue box)

private struct InteractionOverlayRepresentable: UIViewRepresentable {
    var tool: EditorTool
    var eraserRadius: Int
    var scale: CGFloat
    var offset: CGSize
    var fit: CGSize
    var imageSize: CGSize
    var viewSize: CGSize
    var onPan: (CGSize) -> Void
    var onTap: (Int, Int) -> Void
    var onMarqueeSelect: (CGRect) -> Void
    var onEraseStroke: ([(x: Int, y: Int)]) -> Void

    func makeUIView(context: Context) -> InteractionOverlayView {
        let view = InteractionOverlayView()
        view.isMultipleTouchEnabled = false
        view.backgroundColor = .clear
        view.onPan = onPan
        view.onTap = onTap
        view.onMarqueeSelect = onMarqueeSelect
        view.onEraseStroke = onEraseStroke
        return view
    }

    func updateUIView(_ uiView: InteractionOverlayView, context: Context) {
        let toolChanged = uiView.tool != tool
        uiView.tool = tool
        uiView.eraserRadius = eraserRadius
        uiView.canvasScale = scale
        uiView.canvasOffset = offset
        uiView.fit = fit
        uiView.imageSize = imageSize
        uiView.onPan = onPan
        uiView.onTap = onTap
        uiView.onMarqueeSelect = onMarqueeSelect
        uiView.onEraseStroke = onEraseStroke
        // Don't thrash the resting glyph every SwiftUI tick — only on tool change / idle.
        if toolChanged || !uiView.isActivelyDragging {
            uiView.refreshRestingCursor()
        }
    }
}

private final class InteractionOverlayView: UIView {
    var tool: EditorTool = .select
    var eraserRadius: Int = 14
    var canvasScale: CGFloat = 1
    var canvasOffset: CGSize = .zero
    var fit: CGSize = .zero
    var imageSize: CGSize = .zero

    var onPan: ((CGSize) -> Void)?
    var onTap: ((Int, Int) -> Void)?
    var onMarqueeSelect: ((CGRect) -> Void)?
    var onEraseStroke: (([(x: Int, y: Int)]) -> Void)?

    private let cursorView = UIImageView()
    private let marqueeLayer = CAShapeLayer()
    private let eraserRing = CAShapeLayer()
    private let eraserFill = CAShapeLayer()

    private var touchStart: CGPoint?
    private var lastPan: CGPoint?
    private var eraseImagePoints: [(x: Int, y: Int)] = []
    private var isDragging = false
    /// Exposed so SwiftUI updates don't reset graphics mid-gesture.
    var isActivelyDragging: Bool { isDragging }

    override init(frame: CGRect) {
        super.init(frame: frame)
        isOpaque = false
        isUserInteractionEnabled = true

        let config = UIImage.SymbolConfiguration(pointSize: 28, weight: .bold)
        cursorView.image = UIImage(systemName: "cursorarrow", withConfiguration: config)?
            .withTintColor(.white, renderingMode: .alwaysOriginal)
        cursorView.contentMode = .scaleAspectFit
        cursorView.layer.shadowColor = UIColor.black.cgColor
        cursorView.layer.shadowOpacity = 0.55
        cursorView.layer.shadowRadius = 1.5
        cursorView.layer.shadowOffset = CGSize(width: 1, height: 1)
        cursorView.frame = CGRect(x: 0, y: 0, width: 32, height: 32)
        cursorView.isHidden = true
        addSubview(cursorView)

        marqueeLayer.fillColor = UIColor(red: 0.20, green: 0.55, blue: 1.00, alpha: 0.16).cgColor
        marqueeLayer.strokeColor = UIColor(red: 0.20, green: 0.55, blue: 1.00, alpha: 1).cgColor
        marqueeLayer.lineWidth = 2
        marqueeLayer.isHidden = true
        layer.addSublayer(marqueeLayer)

        eraserFill.fillColor = UIColor.orange.withAlphaComponent(0.18).cgColor
        eraserFill.isHidden = true
        layer.addSublayer(eraserFill)

        eraserRing.fillColor = UIColor.clear.cgColor
        eraserRing.strokeColor = UIColor.orange.cgColor
        eraserRing.lineWidth = 2
        eraserRing.isHidden = true
        layer.addSublayer(eraserRing)
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    func refreshRestingCursor() {
        guard !isDragging else { return }
        hideCursor()
        hideEraser()
        marqueeLayer.isHidden = true
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        if !isDragging { refreshRestingCursor() }
    }

    // MARK: - Touches (immediate CALayer updates)

    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        UIApplication.shared.sendAction(
            #selector(UIResponder.resignFirstResponder),
            to: nil,
            from: nil,
            for: nil
        )
        guard let touch = touches.first else { return }
        let p = touch.location(in: self)
        touchStart = p
        lastPan = p
        isDragging = true
        eraseImagePoints = []

        switch tool {
        case .hand:
            hideCursor()
            hideEraser()
            marqueeLayer.isHidden = true
        case .select:
            // Blue box opens immediately (GlassBox-style).
            showCursor(at: p, dimmed: false)
            hideEraser()
            updateMarquee(from: p, to: p)
            marqueeLayer.isHidden = false
        case .eraser:
            hideCursor()
            showEraser(at: p, dimmed: false)
            marqueeLayer.isHidden = true
            if let ip = imagePoint(from: p) {
                eraseImagePoints = [ip]
            }
        }
    }

    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let touch = touches.first, let start = touchStart else { return }
        let p = touch.location(in: self)

        switch tool {
        case .hand:
            if let last = lastPan {
                onPan?(CGSize(width: p.x - last.x, height: p.y - last.y))
            }
            lastPan = p
        case .select:
            showCursor(at: p, dimmed: false)
            updateMarquee(from: start, to: p)
            marqueeLayer.isHidden = false
        case .eraser:
            showEraser(at: p, dimmed: false)
            if let ip = imagePoint(from: p) {
                if let last = eraseImagePoints.last {
                    let dx = ip.x - last.x
                    let dy = ip.y - last.y
                    if dx * dx + dy * dy >= 9 {
                        eraseImagePoints.append(ip)
                    }
                } else {
                    eraseImagePoints.append(ip)
                }
            }
        }
    }

    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let touch = touches.first, let start = touchStart else {
            endGestureCleanup()
            return
        }
        let p = touch.location(in: self)
        let moved = hypot(p.x - start.x, p.y - start.y)

        switch tool {
        case .hand:
            break
        case .select:
            marqueeLayer.isHidden = true
            if moved >= 4 {
                if let rect = imageRect(from: start, to: p) {
                    onMarqueeSelect?(rect)
                }
            } else if let ip = imagePoint(from: p) {
                onTap?(ip.x, ip.y)
            }
            showCursor(at: p, dimmed: false)
        case .eraser:
            if let ip = imagePoint(from: p) {
                eraseImagePoints.append(ip)
            }
            let points = eraseImagePoints
            eraseImagePoints = []
            if !points.isEmpty {
                onEraseStroke?(points)
            }
            showEraser(at: p, dimmed: false)
        }

        endGestureCleanup()
    }

    override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) {
        marqueeLayer.isHidden = true
        eraseImagePoints = []
        endGestureCleanup()
    }

    private func endGestureCleanup() {
        isDragging = false
        touchStart = nil
        lastPan = nil
        hideCursor()
        hideEraser()
        marqueeLayer.isHidden = true
    }

    // MARK: - Draw helpers (no animation — set path immediately)

    private func showCursor(at point: CGPoint, dimmed: Bool) {
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        cursorView.isHidden = false
        // Tip of arrow near the touch point
        cursorView.center = CGPoint(x: point.x + 10, y: point.y + 12)
        cursorView.alpha = dimmed ? 0.45 : 1
        cursorView.layer.removeAllAnimations()
        CATransaction.commit()
    }

    private func hideCursor() {
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        cursorView.isHidden = true
        CATransaction.commit()
    }

    private func showEraser(at point: CGPoint, dimmed: Bool) {
        let r = brushScreenRadius()
        let rect = CGRect(x: point.x - r, y: point.y - r, width: r * 2, height: r * 2)
        let path = UIBezierPath(ovalIn: rect).cgPath
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        eraserFill.path = path
        eraserFill.opacity = Float(dimmed ? 0.35 : 1)
        eraserFill.isHidden = false
        eraserRing.path = path
        eraserRing.opacity = Float(dimmed ? 0.45 : 1)
        eraserRing.isHidden = false
        CATransaction.commit()
    }

    private func hideEraser() {
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        eraserFill.isHidden = true
        eraserRing.isHidden = true
        CATransaction.commit()
    }

    private func updateMarquee(from start: CGPoint, to end: CGPoint) {
        let rect = CGRect(
            x: min(start.x, end.x),
            y: min(start.y, end.y),
            width: max(abs(end.x - start.x), 4),
            height: max(abs(end.y - start.y), 4)
        )
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        marqueeLayer.path = UIBezierPath(rect: rect).cgPath
        marqueeLayer.isHidden = false
        CATransaction.commit()
    }

    private func brushScreenRadius() -> CGFloat {
        let px = CGFloat(max(eraserRadius, 1))
        let imageScale = (fit.width / max(imageSize.width, 1)) * canvasScale
        return max(8, px * imageScale)
    }

    private func imagePoint(from location: CGPoint) -> (x: Int, y: Int)? {
        guard canvasScale > 0, fit.width > 0, fit.height > 0,
              imageSize.width > 0, imageSize.height > 0,
              bounds.width > 0, bounds.height > 0
        else { return nil }

        let center = CGPoint(x: bounds.midX, y: bounds.midY)
        let unscaled = CGPoint(
            x: (location.x - center.x - canvasOffset.width) / canvasScale + center.x,
            y: (location.y - center.y - canvasOffset.height) / canvasScale + center.y
        )
        let origin = CGPoint(
            x: (bounds.width - fit.width) / 2,
            y: (bounds.height - fit.height) / 2
        )
        let local = CGPoint(x: unscaled.x - origin.x, y: unscaled.y - origin.y)
        guard local.x >= 0, local.y >= 0, local.x <= fit.width, local.y <= fit.height else {
            return nil
        }
        let x = Int((local.x / fit.width) * imageSize.width)
        let y = Int((local.y / fit.height) * imageSize.height)
        return (
            min(max(0, x), Int(imageSize.width) - 1),
            min(max(0, y), Int(imageSize.height) - 1)
        )
    }

    private func imageRect(from start: CGPoint, to end: CGPoint) -> CGRect? {
        guard let a = imagePoint(from: start), let b = imagePoint(from: end) else { return nil }
        return CGRect(
            x: CGFloat(min(a.x, b.x)),
            y: CGFloat(min(a.y, b.y)),
            width: CGFloat(max(abs(a.x - b.x), 1)),
            height: CGFloat(max(abs(a.y - b.y), 1))
        )
    }
}
